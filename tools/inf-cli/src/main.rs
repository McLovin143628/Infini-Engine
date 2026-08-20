//! Infinity Engine command-line tool.
//!
//! Subcommands: `inf new <name>` (scaffold a project from a template),
//! `inf cook --project <dir>` (build a shippable asset pack + manifest),
//! `inf export --project <dir>` (assemble a runnable desktop bundle: renamed
//! player exe + pack + manifest + launch config), `inf pack ls <pack>` (inspect a
//! `.inf_pack`), `inf --version`. `inf bindings` lands with its tooling phase.

use std::path::PathBuf;
use std::process::ExitCode;

use inf_asset::{AssetId, PackReader};
use inf_blueprint::BlueprintClass;
use inf_packager::{
    build_mod_wasm, cook, export, export_android, export_web, generate_mod_crate,
    AndroidExportOptions, CookOptions, ExportOptions, ModBuildOptions, ModBuildOutcome,
    WebExportOptions,
};
use inf_project::{Project, ProjectTemplate};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("inf {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("new") => cmd_new(&args[1..]),
        Some("cook") => cmd_cook(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
        Some("pack") => cmd_pack(&args[1..]),
        Some("gis") => cmd_gis(&args[1..]),
        Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}\n");
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!(
        "inf {} — Infinity Engine CLI\n\n\
         USAGE:\n  \
             inf new <name> [--template <slug>] [--dir <path>]\n  \
             inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]\n  \
             inf cook --mods <class.inf_act> [--out <dir>]\n  \
             inf export --project <dir> [--out <dir>] [--target current|web|android] [--player-bin <path>]\n  \
             inf pack ls <pack.inf_pack>\n  \
             inf gis info <file.shp|.geojson> [--crs <spec>]\n  \
             inf gis plan <file> [--kind <kind>] [--crs <spec>] [--max <n>] \
             [--min-length <m>] [--project <dir> | --level <file.inf_lvl> | \
             --anchor <crs>,<easting>,<northing>[,<height>]]\n  \
             inf --version\n\n\
         TEMPLATES:\n  \
             blank-3d (default), 2d-platformer, first-person, hybrid-2.5d\n\n\
         GIS LAYER KINDS:\n  \
             generic (default), roads, streams, lakes, biomes, buildings, parcels\n",
        env!("CARGO_PKG_VERSION")
    );
}

fn cmd_new(args: &[String]) -> ExitCode {
    let mut name: Option<String> = None;
    let mut template = ProjectTemplate::Blank3d;
    let mut dir = PathBuf::from(".");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--template" | "-t" => {
                i += 1;
                match args.get(i).and_then(|s| ProjectTemplate::from_slug(s)) {
                    Some(t) => template = t,
                    None => {
                        eprintln!(
                            "unknown template (use blank-3d / 2d-platformer / first-person / hybrid-2.5d)"
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--dir" | "-d" => {
                i += 1;
                match args.get(i) {
                    Some(d) => dir = PathBuf::from(d),
                    None => {
                        eprintln!("--dir needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other if !other.starts_with('-') && name.is_none() => name = Some(other.to_string()),
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let Some(name) = name else {
        eprintln!("usage: inf new <name> [--template <slug>] [--dir <path>]");
        return ExitCode::FAILURE;
    };

    match Project::create(&dir, &name, template) {
        Ok(p) => {
            println!(
                "Created {} project \"{}\" at {}",
                template.label(),
                p.name(),
                p.root.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]`, or
/// `inf cook --mods <class.inf_act> [--out <dir>]` (the WASM mod cook target).
fn cmd_cook(args: &[String]) -> ExitCode {
    let mut project: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut roots: Option<Vec<AssetId>> = None;
    let mut mods: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mods" => {
                i += 1;
                match args.get(i) {
                    Some(p) => mods = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--mods needs a path to a `.inf_act` blueprint class");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--project" | "-p" => {
                i += 1;
                match args.get(i) {
                    Some(p) => project = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--project needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--out" | "-o" => {
                i += 1;
                match args.get(i) {
                    Some(p) => out = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--out needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--roots" | "-r" => {
                i += 1;
                match args.get(i) {
                    Some(list) => {
                        let mut ids = Vec::new();
                        for tok in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                            match tok.parse::<AssetId>() {
                                Ok(id) => ids.push(id),
                                Err(_) => {
                                    eprintln!("invalid root guid: {tok}");
                                    return ExitCode::FAILURE;
                                }
                            }
                        }
                        roots = Some(ids);
                    }
                    None => {
                        eprintln!("--roots needs a comma-separated guid list");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    // The `--mods` path is a self-contained cook target (blueprint → wasm mod).
    if let Some(class_path) = mods {
        return cmd_cook_mods(&class_path, out);
    }

    let Some(project) = project else {
        eprintln!(
            "usage: inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]\n   \
                    or: inf cook --mods <class.inf_act> [--out <dir>]"
        );
        return ExitCode::FAILURE;
    };
    // Default output: `<project>/Build`.
    let out = out.unwrap_or_else(|| project.join("Build"));

    let opts = CookOptions {
        roots,
        ..Default::default()
    };
    match cook(&project, &out, &opts) {
        Ok(report) => {
            print!("{}", report.render());
            // C4-40: a cook that produced an unbootable pack — or one missing
            // content its levels name — is not a success. It used to print the
            // advisory and exit 0, so CI shipped it green.
            if report.has_blocking() {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("cook failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `inf cook --mods <class.inf_act> [--out <dir>]` — the WASM mod cook target:
/// transpile the blueprint class → Rust, generate the mod `cdylib` crate, and
/// build it to `wasm32-unknown-unknown` when the toolchain is present.
fn cmd_cook_mods(class_path: &PathBuf, out: Option<PathBuf>) -> ExitCode {
    let text = match std::fs::read_to_string(class_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", class_path.display());
            return ExitCode::FAILURE;
        }
    };
    let class: BlueprintClass = match serde_json::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "{} is not a JSON blueprint class: {e}",
                class_path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    let out_dir = out.unwrap_or_else(|| PathBuf::from("Mods"));
    // The generated crate references the shim by a repo-relative path; when
    // writing under an arbitrary out dir, point it at the workspace crate.
    let opts = ModBuildOptions::default();
    let generated = match generate_mod_crate(&class, &opts) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mod codegen failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let crate_dir = out_dir.join(&generated.crate_name);
    if let Err(e) = generated.write_to(&crate_dir) {
        eprintln!("writing mod crate: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "Generated mod crate `{}` at {}",
        generated.crate_name,
        crate_dir.display()
    );

    match build_mod_wasm(&crate_dir) {
        Ok(ModBuildOutcome::Built(wasm)) => {
            println!("Built {}", wasm.display());
            ExitCode::SUCCESS
        }
        Ok(ModBuildOutcome::ToolchainMissing(instructions)) => {
            println!("{instructions}");
            // Not a failure: the crate is generated + ready to build.
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("mod build failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `inf export --project <dir> [--out <dir>] [--target current] [--player-bin <path>]`
fn cmd_export(args: &[String]) -> ExitCode {
    let mut project: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut player_bin: Option<PathBuf> = None;
    let mut target = String::from("current");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" | "-p" => {
                i += 1;
                match args.get(i) {
                    Some(p) => project = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--project needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--out" | "-o" => {
                i += 1;
                match args.get(i) {
                    Some(p) => out = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--out needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--player-bin" => {
                i += 1;
                match args.get(i) {
                    Some(p) => player_bin = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("--player-bin needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--target" => {
                i += 1;
                match args.get(i) {
                    Some(t) => target = t.clone(),
                    None => {
                        eprintln!("--target needs a value (current | web | android)");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let Some(project) = project else {
        eprintln!(
            "usage: inf export --project <dir> [--out <dir>] [--target current|web|android] [--player-bin <path>]"
        );
        return ExitCode::FAILURE;
    };

    match target.as_str() {
        "current" => {
            let opts = ExportOptions {
                out_dir: out,
                player_bin,
                ..Default::default()
            };
            report_export(export(&project, &opts).map(|r| (r.render(), r.has_blocking())))
        }
        // Web (P14.2): cook + wasm bundle skeleton; runs the two-step wasm build
        // when the toolchain is present, else leaves WEB_BUILD.txt instructions.
        "web" => {
            let opts = WebExportOptions {
                out_dir: out,
                run_toolchain: true,
            };
            report_export(export_web(&project, &opts).map(|r| (r.render(), r.has_blocking())))
        }
        // Android (P14.1): cook + cargo-ndk/APK build steps (NDK required).
        "android" => {
            let opts = AndroidExportOptions { out_dir: out };
            report_export(export_android(&project, &opts).map(|r| (r.render(), r.has_blocking())))
        }
        other => {
            eprintln!("unsupported --target '{other}' (current | web | android)");
            ExitCode::FAILURE
        }
    }
}

/// Print an export report or its error.
///
/// C4-40: an export whose cook cannot boot — or whose wasm build was attempted
/// and failed — exits non-zero. It used to print the advisory and return
/// `SUCCESS`, which is how CI shipped an unbootable bundle green.
fn report_export(result: Result<(String, bool), inf_packager::CookError>) -> ExitCode {
    match result {
        Ok((rendered, blocking)) => {
            print!("{rendered}");
            if blocking {
                eprintln!("export produced a build that must not ship (see the errors above)");
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("export failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `inf gis …` — the headless half of the GIS import door (IB-3).
///
/// # The same door the wizard uses, and that is the point
///
/// Both verbs go through `inf_gis::import`: `info` is
/// [`inf_gis::probe`](inf_gis::import::probe) and `plan` is
/// [`inf_gis::import_layer`](inf_gis::import::import_layer). Nothing here reads
/// an attribute, applies a cap or names an entity — every one of those decisions
/// is in Ring 0, so the editor's wizard and this binary cannot drift. The
/// `--digest` line is what makes that provable across a process boundary:
/// `tools/inf-cli/tests/cli.rs` runs the real binary and compares its digest
/// against one computed in-process from the same fixture.
fn cmd_gis(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("info") => cmd_gis_info(&args[1..]),
        Some("plan") => cmd_gis_plan(&args[1..]),
        _ => {
            eprintln!(
                "usage: inf gis info <file> [--crs <spec>]\n       \
                 inf gis plan <file> [--kind <kind>] [--crs <spec>] [--max <n>] \
                 [--min-length <m>] [--project <dir> | --level <file> | \
                 --anchor <crs>,<easting>,<northing>[,<height>]]"
            );
            ExitCode::FAILURE
        }
    }
}

/// Everything both GIS verbs parse.
#[derive(Default)]
struct GisArgs {
    path: Option<PathBuf>,
    crs: Option<String>,
    kind: String,
    max: Option<usize>,
    min_length: Option<f64>,
    project: Option<PathBuf>,
    level: Option<PathBuf>,
    anchor: Option<String>,
}

fn parse_gis_args(args: &[String]) -> Result<GisArgs, String> {
    let mut out = GisArgs::default();
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--crs" | "-c" => out.crs = Some(take(&mut i)?),
            "--kind" | "-k" => out.kind = take(&mut i)?,
            "--max" | "-m" => {
                let v = take(&mut i)?;
                out.max = Some(
                    v.parse::<usize>()
                        .map_err(|_| format!("--max needs a whole number, not {v:?}"))?,
                );
            }
            "--min-length" => {
                let v = take(&mut i)?;
                out.min_length =
                    Some(v.parse::<f64>().map_err(|_| {
                        format!("--min-length needs a length in metres, not {v:?}")
                    })?);
            }
            "--project" | "-p" => out.project = Some(PathBuf::from(take(&mut i)?)),
            "--level" | "-l" => out.level = Some(PathBuf::from(take(&mut i)?)),
            "--anchor" | "-a" => out.anchor = Some(take(&mut i)?),
            other if !other.starts_with('-') && out.path.is_none() => {
                out.path = Some(PathBuf::from(other));
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
        i += 1;
    }
    Ok(out)
}

/// `inf gis info <file>` — what is inside a vector source, before importing it.
fn cmd_gis_info(args: &[String]) -> ExitCode {
    let a = match parse_gis_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(path) = a.path else {
        eprintln!("usage: inf gis info <file.shp|.geojson> [--crs <spec>]");
        return ExitCode::FAILURE;
    };
    let probe = match inf_gis::probe(&path, a.crs.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "{} — {} layer {:?}, {} features ({} points, {} polylines, {} polygons)",
        path.display(),
        probe.format,
        probe.layer_name,
        probe.features,
        probe.points,
        probe.polylines,
        probe.polygons
    );
    println!(
        "crs: {} ({}){}",
        probe.crs.spec,
        probe.crs.origin.label(),
        probe
            .crs
            .name
            .as_deref()
            .map(|n| format!(" — {n}"))
            .unwrap_or_default()
    );
    if (probe.crs.vertical_unit_m - 1.0).abs() > 1e-12 {
        println!("vertical unit: {} m per unit", probe.crs.vertical_unit_m);
    }
    if let Some((lat, lon)) = probe.centre_lat_lon {
        println!("centre: {lat:.5}, {lon:.5}");
    }
    if let Some(code) = probe.suggested_anchor_epsg {
        println!("suggested anchor CRS: EPSG:{code}");
    }
    if let Some((lo, hi)) = probe.bounds_source {
        println!(
            "source bounds: ({:.3}, {:.3}) .. ({:.3}, {:.3})",
            lo.x, lo.z, hi.x, hi.z
        );
    }
    println!("fields:");
    for f in &probe.fields {
        println!(
            "  {:<24} {:>7} set, {:>7} numeric{}",
            f.name,
            f.present,
            f.numeric,
            f.sample
                .as_deref()
                .map(|s| format!("  e.g. {s:?}"))
                .unwrap_or_default()
        );
    }
    for s in probe.skipped.iter().take(5) {
        println!("skipped: {s}");
    }
    for adv in &probe.advisories {
        println!("advisory {adv}");
    }
    ExitCode::SUCCESS
}

/// Resolve the geo-anchor a plan is transformed into.
///
/// Three spellings, in the order an author would reach for them: an explicit
/// `--anchor`, a level file, or a project (whose root level is read through
/// `inf_scene` — the same decode-only reader the shipped player uses, so the CLI
/// links no editor crate to find out where a level is on Earth).
fn resolve_anchor(a: &GisArgs) -> Result<inf_math::geo::GeoAnchor, String> {
    if let Some(spec) = &a.anchor {
        let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
        if parts.len() < 3 {
            return Err(format!(
                "--anchor takes <crs>,<easting>,<northing>[,<height>]; got {spec:?}"
            ));
        }
        let num = |s: &str, what: &str| -> Result<f64, String> {
            s.parse::<f64>()
                .map_err(|_| format!("the anchor's {what} must be a number, not {s:?}"))
        };
        let e = num(parts[1], "easting")?;
        let n = num(parts[2], "northing")?;
        let h = match parts.get(3) {
            Some(v) => num(v, "height")?,
            None => 0.0,
        };
        return inf_gis::anchor_at(parts[0], e, n, h, "unknown").map_err(|e| e.to_string());
    }
    let level = match (&a.level, &a.project) {
        (Some(l), _) => l.clone(),
        (None, Some(p)) => {
            let project = inf_project::Project::open(p).map_err(|e| e.to_string())?;
            // Levels are content (IB-7): the levels root resolves under the
            // content root, and the first `.inf_lvl` in name order is the one
            // whose anchor a headless import means. A project with several is
            // told to name one, rather than being guessed at.
            let dir = project.levels_root();
            let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                .map_err(|e| format!("could not read {}: {e}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "inf_lvl"))
                .collect();
            found.sort();
            found.into_iter().next().ok_or_else(|| {
                format!(
                    "the project at {} contains no levels under {}, so there is \
                     nothing that says where it is on Earth. Pass --anchor, or \
                     --level with a path.",
                    p.display(),
                    dir.display()
                )
            })?
        }
        (None, None) => {
            return Err(
                "a GIS import has to be transformed into SOMETHING: pass --anchor \
                 <crs>,<easting>,<northing>, or --level / --project so the level's \
                 own geo-anchor can be read."
                    .to_string(),
            )
        }
    };
    let lvl = inf_scene::RuntimeLevel::load(&level)
        .map_err(|e| format!("could not read {}: {e}", level.display()))?;
    if !lvl.geo.enabled {
        return Err(format!(
            "the level {} has no geo-anchor, so there is no answer to where its \
             origin is on Earth. Set one in the editor's World Settings, or pass \
             --anchor <crs>,<easting>,<northing>.",
            level.display()
        ));
    }
    Ok(lvl.geo)
}

/// `inf gis plan <file>` — what an import of this file would create.
fn cmd_gis_plan(args: &[String]) -> ExitCode {
    let a = match parse_gis_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(path) = a.path.clone() else {
        eprintln!("usage: inf gis plan <file> [--kind <kind>] [--crs <spec>] [--max <n>]");
        return ExitCode::FAILURE;
    };
    let anchor = match resolve_anchor(&a) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let kind = inf_gis::LayerKind::from_label(&a.kind);
    let mut req = inf_gis::ImportRequest::new(path.clone(), kind);
    req.source_crs = a.crs.clone();
    if let Some(m) = a.max {
        req.options.max_entities = m;
    }
    if let Some(l) = a.min_length {
        req.options.min_length_m = l;
    }
    let imported = match inf_gis::import_layer(&req, &anchor) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let plan = &imported.plan;
    println!(
        "{} — {} as {}, crs {} ({})",
        path.display(),
        imported.layer.name,
        kind.label(),
        imported.crs.spec,
        imported.crs.origin.label()
    );
    println!("{}", plan.summary(&imported.layer.name));
    println!("entities: {}", plan.count());
    println!("too-short: {}", plan.too_short);
    println!("unusable: {}", plan.unusable);
    println!("truncated: {}", plan.truncated);
    println!("cap: {}", plan.cap);
    println!("digest: {:016x}", plan.digest());
    for adv in &plan.advisories {
        println!("advisory: {adv}");
    }
    // A truncated import is a build that lost data. The C4-40 law: a report with
    // a blocking finding in it exits non-zero, so a pipeline stops rather than
    // shipping a city with a hard edge.
    if plan.truncated > 0 {
        eprintln!(
            "{} feature(s) were NOT imported because the entity cap is {}. Pass \
             --max {} to take the whole layer.",
            plan.truncated,
            plan.cap,
            plan.count() + plan.truncated
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// `inf pack ls <pack>` — list a pack's index for debugging.
fn cmd_pack(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("ls") => {
            let Some(path) = args.get(1) else {
                eprintln!("usage: inf pack ls <pack.inf_pack>");
                return ExitCode::FAILURE;
            };
            match PackReader::open(&PathBuf::from(path)) {
                Ok(reader) => {
                    println!(
                        "{} — format v{}, {} assets",
                        path,
                        reader.format_version(),
                        reader.len()
                    );
                    println!(
                        "{:<38} {:<18} {:>10} {:>10}  z",
                        "guid", "kind", "stored", "raw"
                    );
                    for e in reader.index() {
                        println!(
                            "{:<38} {:<18} {:>10} {:>10}  {}",
                            e.guid,
                            e.kind.slug(),
                            e.stored_len,
                            e.uncompressed_len,
                            if e.compressed { "zstd" } else { "raw" }
                        );
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("cannot read pack: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("usage: inf pack ls <pack.inf_pack>");
            ExitCode::FAILURE
        }
    }
}
