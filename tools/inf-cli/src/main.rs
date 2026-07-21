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
use inf_packager::{
    cook, export, export_android, export_web, AndroidExportOptions, CookOptions, ExportOptions,
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
             inf export --project <dir> [--out <dir>] [--target current|web|android] [--player-bin <path>]\n  \
             inf pack ls <pack.inf_pack>\n  \
             inf --version\n\n\
         TEMPLATES:\n  \
             blank-3d (default), 2d-platformer, first-person, hybrid-2.5d\n",
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

/// `inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]`
fn cmd_cook(args: &[String]) -> ExitCode {
    let mut project: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut roots: Option<Vec<AssetId>> = None;

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

    let Some(project) = project else {
        eprintln!("usage: inf cook --project <dir> [--out <dir>] [--roots <guid,guid,…>]");
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
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cook failed: {e}");
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
            report_export(export(&project, &opts).map(|r| r.render()))
        }
        // Web (P14.2): cook + wasm bundle skeleton; runs the two-step wasm build
        // when the toolchain is present, else leaves WEB_BUILD.txt instructions.
        "web" => {
            let opts = WebExportOptions {
                out_dir: out,
                run_toolchain: true,
            };
            report_export(export_web(&project, &opts).map(|r| r.render()))
        }
        // Android (P14.1): cook + cargo-ndk/APK build steps (NDK required).
        "android" => {
            let opts = AndroidExportOptions { out_dir: out };
            report_export(export_android(&project, &opts).map(|r| r.render()))
        }
        other => {
            eprintln!("unsupported --target '{other}' (current | web | android)");
            ExitCode::FAILURE
        }
    }
}

/// Print an export report or its error.
fn report_export(result: Result<String, inf_packager::CookError>) -> ExitCode {
    match result {
        Ok(rendered) => {
            print!("{rendered}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("export failed: {e}");
            ExitCode::FAILURE
        }
    }
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
