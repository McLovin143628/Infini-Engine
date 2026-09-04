//! Infini Engine command-line tool.
//!
//! Subcommands: `inf new <name>` (scaffold a project from a template),
//! `inf cook --project <dir>` (build a shippable asset pack + manifest),
//! `inf export --project <dir>` (assemble a runnable desktop bundle: renamed
//! player exe + pack + manifest + launch config), `inf pack ls <pack>` (inspect a
//! `.ipack`), `inf --version`. `inf bindings` lands with its tooling phase.

use std::path::PathBuf;
use std::process::ExitCode;

use inf_asset::{AssetId, EntryPolicy, PackReader, PackWriter};
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
        Some("island") => cmd_island(&args[1..]),
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
        "inf {} — Infini Engine CLI\n\n\
         USAGE:\n  \
             inf new <name> [--template <slug>] [--dir <path>]\n  \
             inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>] \
             [--block-codec raw|lz4|deflate|zstd]\n  \
             inf cook --mods <class.inf_act> [--out <dir>]\n  \
             inf export --project <dir> [--out <dir>] [--target current|web|android] [--player-bin <path>]\n  \
             inf pack ls [--totals] <pack.ipack>\n  \
             inf gis info <file.shp|.geojson> [--crs <spec>]\n  \
             inf gis plan <file> [--kind <kind>] [--crs <spec>] [--max <n>] \
             [--min-length <m>] [--project <dir> | --level <file.inf_lvl> | \
             --anchor <crs>,<easting>,<northing>[,<height>]]\n  \
             inf island plan  --recipe <island.toml>\n  \
             inf island fetch --recipe <island.toml> [--jobs <n>]\n  \
             inf island build --recipe <island.toml> [--out <dir>] [--offline] \
             [--dry-run]\n  \
             inf island route --recipe <island.toml> [--offline]\n  \
             inf --version\n\n\
         TEMPLATES:\n  \
             blank-3d (default), 2d-platformer, first-person, hybrid-2.5d\n\n\
         GIS LAYER KINDS:\n  \
             generic (default), roads, streams, lakes, biomes, buildings, parcels\n\n\
         BLOCK CODEC (per-tile / per-chunk, inside the streaming containers):\n  \
             zstd    — the default; best ratio and the fastest decode\n  \
             lz4     — weaker ratio, far faster decode; the WEB choice\n  \
             deflate — no reason to pick it; kept for the bake-off\n  \
             raw     — do not transcode at all (the pre-IASSET1 pack)\n",
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
    // IASSET1: the per-block codec for the streaming containers. `None` keeps
    // the default (`inf_terrain::COOK_TILE_CODEC`); `raw` turns the transcode
    // off, which is how a before/after ship-size table is produced from one
    // binary and how a bisect isolates it. A WEB-targeted cook wants `lz4`:
    // `zstd` decodes through the pure-Rust `ruzstd` in a browser, 7.3x slower
    // than the C zstd a desktop links.
    let mut terrain_codec: Option<inf_asset::BlockCodec> = None;

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
            "--block-codec" | "--terrain-codec" => {
                i += 1;
                terrain_codec = match args.get(i).map(String::as_str) {
                    Some("raw") => Some(inf_asset::BlockCodec::Raw),
                    Some("lz4") => Some(inf_asset::BlockCodec::Lz4),
                    Some("deflate") => Some(inf_asset::BlockCodec::Deflate),
                    Some("zstd") => Some(inf_asset::BlockCodec::Zstd),
                    _ => {
                        eprintln!("--block-codec needs one of raw|lz4|deflate|zstd");
                        return ExitCode::FAILURE;
                    }
                };
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
            // `--block-codec` is named here because it is the wave's only
            // web-targeting escape hatch: `zstd` decodes through the pure-Rust
            // `ruzstd` in a browser, 7.3× slower than the C `zstd` a desktop
            // links, and a knob nobody can find is a knob nobody uses.
            "usage: inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]\n   \
                    [--block-codec raw|lz4|deflate|zstd] (default zstd; `raw` = do\n   \
                    not transcode, the pre-IASSET1 pack; `lz4` for a web target)\n   \
                    or: inf cook --mods <class.inf_act> [--out <dir>]"
        );
        return ExitCode::FAILURE;
    };
    // Default output: `<project>/Build`.
    let out = out.unwrap_or_else(|| project.join("Build"));

    let mut opts = CookOptions {
        roots,
        ..Default::default()
    };
    if let Some(codec) = terrain_codec {
        // `raw` means "do not transcode at all" rather than "transcode to raw":
        // the two differ on an asset that arrived compressed, and only the first
        // reproduces the pre-IASSET1 pack byte for byte.
        let c = (codec != inf_asset::BlockCodec::Raw).then_some(codec);
        opts.compression.terrain = c;
        opts.compression.voxel = c;
    }
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

/// `inf island …` — build the island from its recipe (wave I7).
///
/// # THE ONE COMMAND, and what is in it and what is not
///
/// `inf island build --recipe <island.toml>` is the documented command that
/// turns the committed generator into the gigabytes it describes: it plans the
/// source tiles, **fetches the ones the cache lacks**, samples real elevation,
/// carves the designed coastline, derives the water and the biomes, drapes and
/// audits the roads, builds the pyramid and writes the `.inf_terrain`, the road
/// mesh and the `.inf_biomes` into a project's `Content`.
///
/// **The fetch is the only thing in here that is not in Ring 0**, and that split
/// is the whole design. `inf_island::plan_tiles` decides *which* tiles;
/// `inf_island::cache_path` and `inf_island::tile_url` decide *where they live*
/// and *what to ask for*; this binary runs the transfer. So the engine never
/// makes a network call, CI runs every other step against committed bytes, and
/// `--offline` is a refusal rather than a different code path.
///
/// The transfer shells out to `curl`, which is the same ruling `commands/git.rs`
/// took in Phase 5: a subprocess over a CLI every developer machine and every CI
/// runner already has, against linking an HTTPS stack whose root-certificate
/// crate is off this project's licence allow-list (the `ureq`/`webpki-roots`
/// refusal, still standing).
///
/// # C4-40
///
/// A report with a blocking finding **exits non-zero**. An island build is a
/// twenty-minute operation and an advisory printed into a pipeline nobody reads
/// the status of is an advisory nobody reads.
fn cmd_island(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("plan") => cmd_island_plan(&args[1..]),
        Some("fetch") => cmd_island_fetch(&args[1..]),
        Some("build") => cmd_island_build(&args[1..], false),
        Some("route") => cmd_island_build(&args[1..], true),
        _ => {
            eprintln!(
                "usage: inf island plan  --recipe <island.toml>\n       \
                 inf island fetch --recipe <island.toml> [--jobs <n>]\n       \
                 inf island build --recipe <island.toml> [--out <dir>] \
                 [--offline] [--dry-run]\n       \
                 inf island route --recipe <island.toml> [--offline]"
            );
            ExitCode::FAILURE
        }
    }
}

/// Everything the island verbs parse.
#[derive(Default)]
struct IslandArgs {
    recipe: Option<PathBuf>,
    out: Option<PathBuf>,
    jobs: usize,
    offline: bool,
    dry_run: bool,
}

fn parse_island_args(args: &[String]) -> Result<IslandArgs, String> {
    let mut out = IslandArgs {
        jobs: 8,
        ..Default::default()
    };
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--recipe" | "-r" => out.recipe = Some(PathBuf::from(take(&mut i)?)),
            "--out" | "-o" => out.out = Some(PathBuf::from(take(&mut i)?)),
            "--jobs" | "-j" => {
                let v = take(&mut i)?;
                out.jobs = v
                    .parse::<usize>()
                    .map_err(|_| format!("--jobs needs a whole number, not {v:?}"))?
                    .clamp(1, 64);
            }
            "--offline" => out.offline = true,
            "--dry-run" => out.dry_run = true,
            other if !other.starts_with('-') && out.recipe.is_none() => {
                out.recipe = Some(PathBuf::from(other));
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
        i += 1;
    }
    Ok(out)
}

fn island_recipe(a: &IslandArgs) -> Result<inf_island::IslandRecipe, String> {
    let p = a
        .recipe
        .as_ref()
        .ok_or_else(|| "no recipe: pass --recipe <island.toml>".to_string())?;
    inf_island::IslandRecipe::load(p).map_err(|e| e.to_string())
}

/// `inf island plan` — what the recipe will ask the network for, before it does.
fn cmd_island_plan(args: &[String]) -> ExitCode {
    let a = match parse_island_args(args).and_then(|a| island_recipe(&a).map(|r| (a, r))) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let (a, recipe) = a;
    let plan = match inf_island::plan_tiles(&recipe) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let cache = recipe.cache_dir();
    let missing = plan.missing_in(&cache);
    println!("island:      {}", recipe.name);
    println!(
        "world:       {} x {} level-0 tiles of {}^2 at {} m = {:.0} x {:.0} m ({:.2} km2)",
        recipe.grid.tiles,
        recipe.grid.tiles,
        recipe.grid.tile_resolution,
        recipe.grid.meters_per_sample,
        recipe.grid.extent_m(),
        recipe.grid.extent_m(),
        recipe.grid.extent_m() * recipe.grid.extent_m() / 1.0e6
    );
    println!("samples:     {}", recipe.grid.sample_count());
    println!(
        "extent:      lon {:.5}..{:.5}, lat {:.5}..{:.5}",
        plan.lon.0, plan.lon.1, plan.lat.0, plan.lat.1
    );
    println!(
        "source:      {} tiles at z{} = {:.3} m/px of ground against a {:.3} m \
         grid ({:.2}x upsample)",
        plan.len(),
        plan.zoom,
        plan.ground_m_per_px,
        plan.grid_m_per_sample,
        plan.upsample_ratio()
    );
    println!("cache:       {}", cache.display());
    println!("to fetch:    {} of {}", missing.len(), plan.len());
    if !missing.is_empty() {
        println!(
            "first:       {}",
            inf_island::tile_url(&recipe.source.url, missing[0])
        );
    }
    let _ = a;
    ExitCode::SUCCESS
}

/// `inf island fetch` — fill the cache. **The one network step in this repo.**
fn cmd_island_fetch(args: &[String]) -> ExitCode {
    let a = match parse_island_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let recipe = match island_recipe(&a) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match fetch_tiles(&recipe, a.jobs, false) {
        Ok(n) => {
            println!("fetched {n} tiles into {}", recipe.cache_dir().display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// Fetch every tile the plan names that the cache lacks.
///
/// # The two things a downloader must not do
///
/// **It must not keep what it was given.** The endpoint answers `NoSuchKey` with
/// a 299-byte XML body and HTTP 200-shaped plumbing will happily write it to
/// `15/5179/11205.png`. A cached error page decodes as nothing, which is a flat
/// plain where a mountain is — so the response is checked for a PNG signature
/// and refused by name if it is not one, before it reaches the cache.
///
/// **It must not half-write.** The transfer goes to a `.part` beside the target
/// and is renamed on success, so an interrupted fetch leaves no file rather than
/// a truncated one the next build would trust.
fn fetch_tiles(
    recipe: &inf_island::IslandRecipe,
    jobs: usize,
    quiet: bool,
) -> Result<usize, String> {
    let plan = inf_island::plan_tiles(recipe).map_err(|e| e.to_string())?;
    let cache = recipe.cache_dir();
    let missing = plan.missing_in(&cache);
    if missing.is_empty() {
        if !quiet {
            println!("every one of the {} tiles is already cached", plan.len());
        }
        return Ok(0);
    }
    if !quiet {
        println!(
            "fetching {} of {} tiles at z{} into {}",
            missing.len(),
            plan.len(),
            plan.zoom,
            cache.display()
        );
    }
    let done = std::sync::atomic::AtomicUsize::new(0);
    let failed: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let chunk = missing.len().div_ceil(jobs.max(1));
    std::thread::scope(|s| {
        for part in missing.chunks(chunk.max(1)) {
            let done = &done;
            let failed = &failed;
            let cache = &cache;
            let url = &recipe.source.url;
            s.spawn(move || {
                for t in part {
                    match fetch_one(url, *t, cache) {
                        Ok(()) => {
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            if !quiet && n.is_multiple_of(25) {
                                println!("  {n} fetched");
                            }
                        }
                        Err(e) => failed.lock().expect("fetch report").push(e),
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().expect("fetch report");
    if !failed.is_empty() {
        return Err(format!(
            "{} of {} tiles could not be fetched; the first is:\n  {}",
            failed.len(),
            missing.len(),
            failed[0]
        ));
    }
    Ok(done.into_inner())
}

/// PNG's own eight-byte signature. A response that does not start with it is not
/// a tile, whatever the transfer said.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

fn fetch_one(
    url_template: &str,
    t: inf_island::TileId,
    cache: &std::path::Path,
) -> Result<(), String> {
    let url = inf_island::tile_url(url_template, t);
    let target = inf_island::cache_path(cache, t);
    let dir = target
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", target.display()))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let part = target.with_extension("part");
    let out = std::process::Command::new("curl")
        .arg("-sS")
        .arg("--fail")
        .arg("--retry")
        .arg("3")
        .arg("--retry-delay")
        .arg("1")
        .arg("-o")
        .arg(&part)
        .arg(&url)
        .output()
        .map_err(|e| {
            format!(
                "could not run `curl` for {url}: {e}. The island fetch shells out \
                 to curl rather than linking an HTTPS stack — see the ruling on \
                 `inf island`."
            )
        })?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "{url}: curl exited {} — {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let bytes = std::fs::read(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    if bytes.len() < PNG_MAGIC.len() || bytes[..PNG_MAGIC.len()] != PNG_MAGIC {
        let head = String::from_utf8_lossy(&bytes[..bytes.len().min(120)]).to_string();
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "{url} answered {} bytes that are not a PNG: {head:?}. A dataset that \
             does not have a tile at this zoom answers with an error document, \
             and caching one would build a flat plain where a mountain is.",
            bytes.len()
        ));
    }
    std::fs::rename(&part, &target)
        .map_err(|e| format!("{} -> {}: {e}", part.display(), target.display()))?;
    Ok(())
}

/// `inf island build` (and `inf island route`, which is the same build with the
/// road planner switched on).
fn cmd_island_build(args: &[String], replan_roads: bool) -> ExitCode {
    let a = match parse_island_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let recipe = match island_recipe(&a) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if !a.offline {
        if let Err(e) = fetch_tiles(&recipe, a.jobs, false) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }
    let started = std::time::Instant::now();
    // **`route` is TWO passes, and it has to be.** The first plans the network
    // against the ground as it stands and writes the layer; the second reads that
    // layer back, levels the road corridor into the terrain, and audits the
    // ground the road will actually sit on. Auditing after only the first pass
    // measures a road nobody has built yet — 8.11 % over the ceiling against the
    // 0.29 % the finished one holds.
    if replan_roads {
        match inf_island::build_island(&recipe, &inf_island::BuildOptions::planning_pass()) {
            Ok(b) => println!(
                "[    route] planned {} links, {:.2} km; re-building against them",
                b.routes.len(),
                b.report
                    .roads
                    .total_km
                    .max(b.routes.iter().map(|r| r.length_m()).sum::<f64>() / 1000.0)
            ),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let opts = inf_island::BuildOptions {
        rederive_layers: false,
        replan_roads: false,
        dry_run: a.dry_run,
    };
    let build = match inf_island::build_island(&recipe, &opts) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    for l in &build.log {
        println!("[{:>9}] {}", l.step.label(), l.note);
    }
    print!("{}", build.report.summary());
    let start = build.player_start();
    println!(
        "  start      ({:.1}, {:.1}, {:.1})",
        start.x, start.y, start.z
    );
    println!("  elapsed    {:.1} s", started.elapsed().as_secs_f64());

    if !a.dry_run {
        // **The default output is OUTSIDE the tree**, beside the tile cache the
        // recipe already points out of it. A third of a gigabyte of terrain
        // landing under `samples/` is a build artifact in a source tree, and the
        // first run of this command put it there.
        let out = a.out.clone().unwrap_or_else(|| {
            recipe
                .cache_dir()
                .parent()
                .map(|p| p.join("project"))
                .unwrap_or_else(|| recipe.base_dir.join("build"))
        });
        let content = out.join("Content");
        match inf_island::write_content(&build, &content) {
            Ok(files) => {
                for f in &files {
                    println!("  wrote      {f}");
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
        if let Err(e) = inf_project::ProjectManifest::new(&recipe.name, "blank-3d").save(&out) {
            eprintln!(
                "could not scaffold the island project at {}: {e}",
                out.display()
            );
            return ExitCode::FAILURE;
        }
        println!("  project    {}", out.display());
        // Wave CERT1: the built project is what the editor's THIRD boot rung
        // looks for (`inf_project::boot::find_showcase` walks up from the
        // executable for `island-build/project`). Saying so here is the only
        // place a reader learns that building the island is what makes the
        // application open on it — and the only place to learn the override on
        // a machine whose layout the walk cannot reach.
        // The indent is a WIDTH SPECIFIER and not spaces in the literal: a run
        // of six or more spaces inside a string is what an eaten
        // backslash-continuation leaves behind, `cook`'s own source gate bans
        // the shape, and it cannot tell a deliberate indent from the defect --
        // correctly, since it has caught six real ones.
        println!(
            "  {:<11}the editor opens this project on launch once it is built here",
            "boot"
        );
        println!(
            "  {:<11}set {}=<project root> to point it somewhere else",
            "",
            inf_project::BOOT_PROJECT_ENV
        );
    }

    // C4-40: a blocking finding exits non-zero.
    if !build.report.is_clean() {
        eprintln!(
            "\nthe island has {} blocking finding(s); the first is: {}",
            build.report.blocking.len(),
            build
                .report
                .blocking
                .first()
                .map(String::as_str)
                .unwrap_or("")
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
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
            // The path is the first NON-FLAG argument, not `args[1]` — which is
            // what `--totals` would otherwise be read as, producing an ENOENT
            // naming a flag.
            let Some(path) = args[1..].iter().find(|a| !a.starts_with("--")) else {
                eprintln!("usage: inf pack ls [--totals] <pack.ipack>");
                return ExitCode::FAILURE;
            };
            // `--totals` aggregates by kind instead of listing entries: the
            // ship-size table, from the ONE producer a `CookReport`'s
            // `kind_bytes` also reads (IASSET1). An entry list of 1 100 assets
            // cannot be read as a size table, and a size table assembled by eye
            // from one is how two numbers for one quantity get into a memo.
            let totals = args.iter().any(|a| a == "--totals");
            match PackReader::open(&PathBuf::from(path)) {
                Ok(reader) => {
                    println!(
                        "{} — format v{}, {} assets",
                        path,
                        reader.format_version(),
                        reader.len()
                    );
                    if totals {
                        println!(
                            "{:<18} {:>7} {:>16} {:>16} {:>8}",
                            "kind", "n", "stored", "raw", "ratio"
                        );
                        let (mut ts, mut tr, mut tn) = (0u64, 0u64, 0usize);
                        let mut rows: Vec<_> = reader.kind_totals().into_iter().collect();
                        rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.stored_bytes));
                        for (kind, v) in rows {
                            println!(
                                "{:<18} {:>7} {:>16} {:>16} {:>8.3}",
                                kind,
                                v.count,
                                v.stored_bytes,
                                v.uncompressed_bytes,
                                v.ratio()
                            );
                            ts += v.stored_bytes;
                            tr += v.uncompressed_bytes;
                            tn += v.count;
                        }
                        let ratio = if tr == 0 { 1.0 } else { ts as f64 / tr as f64 };
                        println!("{:<18} {tn:>7} {ts:>16} {tr:>16} {ratio:>8.3}", "TOTAL");
                        // The file is bigger than the blobs: header, index and
                        // 16-byte alignment padding. Stating the gap is what
                        // stops "the kinds add up to less than the pack" being
                        // read as a missing kind.
                        if let Ok(meta) = std::fs::metadata(path) {
                            println!(
                                "file {} B; index + padding {} B",
                                meta.len(),
                                meta.len().saturating_sub(ts)
                            );
                        }
                        // **A 1.000 in this table does not mean "not compressed"**
                        // for a streaming kind, and a reader who takes it that way
                        // reaches the wrong conclusion about where the download
                        // went. `raw` is the pack ENTRY's `uncompressed_len`, and
                        // a `BlockCompressed` kind ships its entry raw on purpose
                        // — its saving already happened per block, inside the
                        // payload, before this file existed. Said by the tool that
                        // prints the table rather than only in the memo that
                        // quotes it.
                        if reader.index().any(|e| {
                            PackWriter::entry_policy(e.kind) == EntryPolicy::BlockCompressed
                        }) {
                            println!(
                                "note: a streaming kind (terrain, voxel, partition) ships its \
                                 ENTRY raw, so its ratio here is 1.000 by design — the \
                                 per-block saving is inside the payload and is only visible \
                                 by comparing two packs."
                            );
                        }
                        return ExitCode::SUCCESS;
                    }
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
            eprintln!("usage: inf pack ls [--totals] <pack.ipack>");
            ExitCode::FAILURE
        }
    }
}
