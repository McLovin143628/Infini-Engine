//! **The InfiniScript API Manual, generated from the verb registry.**
//!
//! The direction memo asks for a manual and says why it must not be written by
//! hand: *"a hand-written one goes stale on the first verb, and the registry
//! already carries names, categories, descriptions and typed pins."* This is
//! that manual's generator, and the arm that keeps the committed page equal to
//! the registry.
//!
//! # It is a DRIFT CHECK, not a regenerator
//!
//! `inf-studio`'s `the_typescript_mirror_is_pinned_to_this_encoder` states the
//! shape this file copies, and its reason: *"a test that regenerates its own
//! expectation is the vacuous shape this campaign has caught eight times."* So
//! the committed page is **compared**, never silently rewritten, and a
//! deliberate move is blessed with `INF_BLESS_API_MANUAL=1`.
//!
//! That is deliberately not the ts-rs bindings shape, which writes on every run
//! and leans on a CI job doing `git diff`. Comparing in process costs no CI job
//! at all: this arm runs inside the ordinary workspace battery on all three
//! operating systems, and a page that stopped matching the registry reddens with
//! a diff rather than with an untracked file somebody has to notice.
//!
//! # What the page is FOR
//!
//! A designer reading `docs/book` learns what they can say. Every callable verb,
//! grouped by namespace, with its arguments, their types, whether each is
//! required, what it answers, and whether it is a statement or a value. The
//! refusals are in it too — the operators, the flow palette, the literals — with
//! the syntax that replaces each, because "why can't I call `math.add`" is the
//! first question the surface produces.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use inf_blueprint::lower::role_of;
use inf_blueprint::nodekit::{NodeRole, EXEC_IN};
use inf_graph::{NodeDef, PortType};
use inf_script::Verbs;

/// Where the generated page lives, relative to this crate.
fn manual_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("book")
        .join("src")
        .join("infiniscript-api.md")
}

/// The banner. Names the generator by the test that owns it, so a reader who
/// edits the page by hand is told where to go instead.
const BANNER: &str = "<!-- GENERATED from the InfiniScript verb registry by \
`cargo test -p inf-script --test api_manual`. Do not hand-edit; re-bless with \
INF_BLESS_API_MANUAL=1. -->";

/// A description as one markdown table cell.
///
/// Two hazards, both structural rather than hypothetical. A `NodeDef`
/// description is an ordinary Rust string, so it may hold a **newline** — a
/// `\n` escape, or a run of them from a doc written as separate lines — and a
/// newline inside a table row ends the table. And a description containing a
/// **pipe** would end the cell. Neither is a thing a verb's author should have
/// to remember, so the renderer takes care of it: whitespace runs collapse to
/// one space, and a pipe is escaped.
fn one_line(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('|', "\\|")
}

/// A port's written type.
fn ty_name(ty: &PortType) -> &'static str {
    match ty {
        PortType::Bool => "bool",
        PortType::Int => "int",
        PortType::Float => "float",
        PortType::Str => "string",
        PortType::Named(_) => "named",
        PortType::Wildcard => "any",
        PortType::Exec => "exec",
    }
}

/// The data (non-exec) input ports of a node, in declaration order — which is
/// argument order, and the one fact the manual most has to get right.
fn data_inputs(def: &NodeDef) -> Vec<(String, &'static str, bool)> {
    def.inputs
        .iter()
        .filter(|p| !p.ty.is_exec() && !p.param_pin)
        .map(|p| (p.name.clone(), ty_name(&p.ty), p.required))
        .collect()
}

fn data_outputs(def: &NodeDef) -> Vec<(String, &'static str)> {
    def.outputs
        .iter()
        .filter(|p| !p.ty.is_exec())
        .map(|p| (p.name.clone(), ty_name(&p.ty)))
        .collect()
}

/// How a call is spelled in `.infini`, arguments and all.
fn signature(def: &NodeDef, outs: &[(String, &'static str)]) -> String {
    let args: Vec<String> = data_inputs(def)
        .into_iter()
        .map(|(n, t, req)| {
            if req {
                format!("{n}: {t}")
            } else {
                format!("[{n}: {t}]")
            }
        })
        .collect();
    // A multi-result pure query names its result as a third segment — the
    // lowerer's rule that one wire carries one scalar.
    let has_exec = def.input(EXEC_IN).is_some();
    let name = if !has_exec && outs.len() > 1 {
        format!("{}.<{}>", def.type_id, outs[0].0)
    } else {
        def.type_id.clone()
    };
    format!("{name}({})", args.join(", "))
}

/// Render the whole manual.
fn render(verbs: &Verbs) -> String {
    let (count, namespaces) = verbs.census();
    let mut out = String::new();
    let _ = writeln!(out, "{BANNER}\n");
    let _ = writeln!(out, "# The InfiniScript API");
    let _ = writeln!(
        out,
        "\nEvery verb an InfiniScript can call, generated from the engine's own \
         verb registry — the same table the Blueprint palette is built from and \
         the same one the parser resolves a call against. If it is not here, a \
         script cannot say it.\n"
    );
    let _ = writeln!(
        out,
        "The surface is **{count} registered nodes across {namespaces} \
         namespaces**. Not all of them are *callable*: the arithmetic and \
         comparison operators, the control-flow palette, the literals, the \
         events and the two member-variable nodes are all written as syntax \
         instead, and the last section lists each with the spelling that \
         replaces it.\n"
    );
    let _ = writeln!(
        out,
        "**How to read a row.** `door.use(x: float, y: float, z: float)` takes \
         three arguments in that order. A name in `[square brackets]` is \
         optional and defaults to zero, false or the empty string. A verb marked \
         *statement* runs for its effect and is written on a line of its own; a \
         verb marked *value* answers something and can go inside an expression. \
         A verb that is both can be used either way — write it as a statement to \
         ignore what it answers.\n"
    );

    // Namespaces in registration order, which is palette order, which is the
    // order somebody chose — never a hash order.
    let mut order: Vec<String> = Vec::new();
    for def in verbs.registry().ordered() {
        let ns = def
            .type_id
            .split('.')
            .next()
            .unwrap_or_default()
            .to_string();
        if !order.contains(&ns) {
            order.push(ns);
        }
    }

    let mut refused: Vec<(String, String)> = Vec::new();
    for ns in &order {
        let mut rows = String::new();
        for def in verbs
            .registry()
            .ordered()
            .filter(|d| d.type_id.split('.').next() == Some(ns.as_str()))
        {
            let outs = data_outputs(def);
            let mut segments: Vec<String> = def.type_id.split('.').map(str::to_string).collect();
            let has_exec = def.input(EXEC_IN).is_some();
            if !has_exec && outs.len() > 1 {
                segments.push(outs[0].0.clone());
            }
            match verbs.resolve(&segments) {
                Ok(_) => {
                    let kind = match (has_exec, outs.is_empty()) {
                        (true, true) => "statement",
                        (true, false) => "statement, value",
                        (false, _) => "value",
                    };
                    let answers = if outs.is_empty() {
                        "—".to_string()
                    } else {
                        outs.iter()
                            .map(|(n, t)| format!("`{n}`: {t}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    let _ = writeln!(
                        rows,
                        "| `{}` | {kind} | {answers} | {} |",
                        signature(def, &outs),
                        one_line(&def.description)
                    );
                }
                Err(e) => {
                    // A refused node is still part of the surface a reader
                    // meets; it goes in the last section with its remedy.
                    if role_of(&def.type_id, has_exec) != NodeRole::Event {
                        refused.push((def.type_id.clone(), e.message()));
                    }
                }
            }
        }
        if rows.is_empty() {
            continue;
        }
        let _ = writeln!(out, "## `{ns}.*`\n");
        let _ = writeln!(out, "| call | kind | answers | what it does |");
        let _ = writeln!(out, "|---|---|---|---|");
        let _ = write!(out, "{rows}");
        out.push('\n');
    }

    let _ = writeln!(out, "## Written as syntax instead\n");
    let _ = writeln!(
        out,
        "These are registered nodes a Blueprint graph draws and InfiniScript \
         spells another way. Calling one is a refusal that names the \
         replacement, so the compiler tells you this table rather than making \
         you find it.\n"
    );
    let _ = writeln!(out, "| node | write instead |");
    let _ = writeln!(out, "|---|---|");
    for (id, why) in &refused {
        let _ = writeln!(out, "| `{id}` | {} |", one_line(why));
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "Events are the other family that is not a call: an event is a \
         handler's header (`on tick(dt) … end`), not something a script \
         invokes."
    );
    out
}

/// **The manual matches the registry.**
#[test]
fn the_api_manual_matches_the_verb_registry() {
    let verbs = Verbs::new();
    let want = render(&verbs);
    let path = manual_path();

    if std::env::var("INF_BLESS_API_MANUAL").is_ok() {
        std::fs::write(&path, &want).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }

    let got = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. The generated API manual is COMMITTED (mdBook's \
             `create-missing = false` means a page named in SUMMARY.md must \
             exist at checkout). Re-bless with INF_BLESS_API_MANUAL=1.",
            path.display()
        )
    });
    // The P22 CRLF law: a Windows checkout under `core.autocrlf = true` reads
    // this file with `\r\n`, and the generator writes `\n`.
    let got = got.replace("\r\n", "\n");

    if got == want {
        return;
    }
    // Name the first line that differs rather than printing two documents.
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let at = g.iter().zip(&w).position(|(a, b)| a != b);
    let detail = match at {
        Some(i) => format!(
            "line {}:\n  committed: {}\n  registry:  {}",
            i + 1,
            g[i],
            w[i]
        ),
        None => format!(
            "the committed page has {} lines and the registry renders {}",
            g.len(),
            w.len()
        ),
    };
    panic!(
        "the committed API manual no longer matches the verb registry — {detail}\n\n\
         If the registry moved on purpose, re-bless:\n    \
         INF_BLESS_API_MANUAL=1 cargo test -p inf-script --test api_manual"
    );
}

/// **The book can build.** `docs/book/book.toml` sets `create-missing = false`,
/// so mdBook refuses rather than stubs a page named in `SUMMARY.md` that is not
/// on disk — and CI's docs job has no Rust toolchain, so it cannot generate one
/// first. A generated page must therefore be **committed**, and the summary must
/// name both it and the hand-written page beside it.
///
/// Checked here because this machine has no `mdbook` and the failure it prevents
/// is a red CI job whose message is about a missing file rather than about the
/// generator that owed it.
#[test]
fn the_summary_names_both_pages_and_both_exist() {
    let book = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("book");
    let summary = std::fs::read_to_string(book.join("src").join("SUMMARY.md"))
        .expect("docs/book/src/SUMMARY.md");
    for page in ["./infiniscript.md", "./infiniscript-api.md"] {
        assert!(
            summary.contains(page),
            "SUMMARY.md does not name {page}, so the book will not link it"
        );
        let file = book.join("src").join(page.trim_start_matches("./"));
        assert!(
            file.is_file(),
            "{} is named in SUMMARY.md and is not on disk; `create-missing = false` \
             makes that a failed `mdbook build`",
            file.display()
        );
    }
    // The language page points at the generated one, which is the only way a
    // reader finds it — and it is the SCRIPT1b carried item this closes: before
    // this wave the book named Infini Blueprints twice and `.infini` never.
    let page = std::fs::read_to_string(book.join("src").join("infiniscript.md"))
        .expect("the language page");
    assert!(
        page.contains("infiniscript-api.md"),
        "the language page must link the generated API manual"
    );
    for from in ["introduction.md", "blueprints-101.md"] {
        let src = std::fs::read_to_string(book.join("src").join(from))
            .unwrap_or_else(|e| panic!("{from}: {e}"));
        assert!(
            src.contains("infiniscript.md"),
            "{from} does not mention InfiniScript, so a reader arriving at the \
             graph face never learns there is a text one"
        );
    }
}

/// **Every verb a script can call carries a doc line**, because the manual's
/// last column is that doc line and a blank one is a blank row.
///
/// Scoped to the *callable* surface on purpose: the operators, the literals and
/// the flow palette are refusals with their own remedy text, and demanding a
/// description of `math.add` would be demanding documentation for `+`.
#[test]
fn every_callable_verb_carries_a_doc_line() {
    let verbs = Verbs::new();
    let mut bare: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for def in verbs.registry().ordered() {
        let outs = data_outputs(def);
        let mut segments: Vec<String> = def.type_id.split('.').map(str::to_string).collect();
        let has_exec = def.input(EXEC_IN).is_some();
        if !has_exec && outs.len() > 1 {
            segments.push(outs[0].0.clone());
        }
        if verbs.resolve(&segments).is_err() {
            continue;
        }
        checked += 1;
        if def.description.trim().is_empty() {
            bare.push(def.type_id.clone());
        }
    }
    assert!(
        checked >= 87,
        "only {checked} verbs were checked — the callable surface collapsed and \
         this arm stopped measuring anything"
    );
    assert!(
        bare.is_empty(),
        "{} callable verb(s) have no description, so the generated API manual \
         has a blank row for each: {bare:?}",
        bare.len()
    );
}

/// **The manual is not a stub.** Every namespace with a callable verb has a
/// section, every callable verb has a row, and the refusal table is populated.
///
/// A drift check compares two things; if the generator ever produced an empty
/// document it would compare an empty document to an empty file and pass.
#[test]
fn the_manual_covers_the_surface_it_claims_to() {
    let verbs = Verbs::new();
    let page = render(&verbs);
    let (count, namespaces) = verbs.census();

    let mut callable = 0usize;
    for def in verbs.registry().ordered() {
        let outs = data_outputs(def);
        let mut segments: Vec<String> = def.type_id.split('.').map(str::to_string).collect();
        if def.input(EXEC_IN).is_none() && outs.len() > 1 {
            segments.push(outs[0].0.clone());
        }
        if verbs.resolve(&segments).is_ok() {
            callable += 1;
            assert!(
                page.contains(&format!("`{}(", def.type_id))
                    || page.contains(&format!("`{}.<", def.type_id)),
                "`{}` is callable and has no row in the manual",
                def.type_id
            );
        }
    }
    assert!(callable >= 87, "only {callable} callable verbs");
    assert!(
        page.contains(&format!("{count} registered nodes across {namespaces}")),
        "the manual must state the census it was generated from"
    );
    // The refusal table earns its place: the operators are the first thing a
    // reader asks about.
    for id in [
        "math.add",
        "cmp.lt",
        "logic.not",
        "flow.branch",
        "lit.float",
    ] {
        assert!(
            page.contains(&format!("| `{id}` |")),
            "`{id}` should be in the written-as-syntax table"
        );
    }
    // And an event must NOT be: it is not a call at all, and listing it under
    // "write instead" would be advice about a thing nobody tried to do.
    assert!(
        !page.contains("| `event.tick` |"),
        "events are a handler's header, not a refused call"
    );
}
