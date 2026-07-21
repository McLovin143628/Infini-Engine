//! Infinity Engine command-line tool.
//!
//! Subcommands: `inf new <name>` (scaffold a project from a template),
//! `inf --version`. `inf cook` (asset packs) and `inf bindings` land with their
//! phases (P9 / tooling).

use std::path::PathBuf;
use std::process::ExitCode;

use inf_project::{Project, ProjectTemplate};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("inf {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("new") => cmd_new(&args[1..]),
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
             inf --version\n\n\
         TEMPLATES:\n  \
             blank-3d (default), 2d-platformer, first-person\n",
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
