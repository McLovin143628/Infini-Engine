//! InfiniScript editor commands (wave SCRIPT2b).
//!
//! One command: [`script_check`], which hands the editor the **Ring-0 parse
//! refusals** for a buffer of `.infini` text. It is the reason the frontend has
//! a tokenizer and not a parser — every semantic claim the editor makes about a
//! script (this line is wrong, this variable is not declared, this call needs
//! three arguments) is produced here, by `inf_script::compile_bytes`, which is
//! the same door the asset watcher, `inf cook` and the PIE payload builder go
//! through. There is exactly one InfiniScript compiler in this system and the
//! editor is not a second one.
//!
//! # Why it takes TEXT and not a path
//!
//! The buffer in front of the author is the program they are asking about, and
//! it is usually dirty. Checking the file on disk would answer a question
//! nobody asked and would need a filesystem door (`super::paths`' confinement
//! rule) for no gain — the file's own compile already happens, on save, in
//! `assets::hot_reload_scripts`, and its diagnostics already reach the Output
//! Log. `path` is carried only as the **label** a byte-level refusal names, so
//! "Door.infini is larger than the 1 MiB source limit" reads like the cook's
//! version of the same sentence.
//!
//! # Refusals are values, all the way out
//!
//! The command's `Err` is for a broken *command*, and there is no way to break
//! this one, so it never returns one. A file that will not compile is an `Ok`
//! carrying its diagnostics — P21's law, kept across the IPC boundary rather
//! than abandoned at it. A file that compiles with warnings is the same `Ok`
//! carrying those.

use inf_editor_core::ipc::ScriptDiagnosticDto;

/// The label a diagnostic names: the file's leaf, or a stand-in for an unsaved
/// buffer. Never the whole path — the editor already knows which tab this is,
/// and a message that repeats an absolute path is a message the panel truncates.
fn label_of(path: Option<&str>) -> String {
    path.and_then(|p| p.rsplit(['/', '\\']).next())
        .filter(|leaf| !leaf.is_empty())
        .unwrap_or("(unsaved script)")
        .to_string()
}

/// Compile `text` through the one file door and answer with what it had to say.
///
/// `compile_bytes` rather than `compile`: the door owns the byte-level contract
/// (the `MAX_SOURCE_BYTES` ceiling, the byte-order mark), so a buffer the cook
/// would refuse is refused here too, in the same words. The text is already a
/// `String`, so the UTF-8 half of that contract is satisfied before we arrive.
pub fn check_text(text: &str, path: Option<&str>) -> Vec<ScriptDiagnosticDto> {
    let label = label_of(path);
    match inf_script::compile_bytes(text.as_bytes(), &label, format!("check:{label}")) {
        Ok((_class, warnings)) => warnings.iter().map(ScriptDiagnosticDto::from).collect(),
        Err(diags) => diags.iter().map(ScriptDiagnosticDto::from).collect(),
    }
}

/// Check a buffer of InfiniScript. `path` is a label only — nothing is read
/// from disk. Always `Ok`; a refusal is a value in the list.
#[tauri::command]
pub async fn script_check(
    text: String,
    path: Option<String>,
) -> Result<Vec<ScriptDiagnosticDto>, String> {
    Ok(check_text(&text, path.as_deref()))
}

/// **The graph↔text bridge's editor half** (wave SCRIPT2b): render a blueprint
/// asset as the `.infini` file it would be.
///
/// `AssetState::load_blueprint_class_result` already reads BOTH kinds — a
/// `.inf_act`'s committed JSON and a `.infini` through the file door — so this
/// is one door for "show me this class as text", whichever it was authored as.
///
/// **Read-only in the editor, and the reason is a real one rather than
/// caution.** Writing the text back would mean re-parsing it into a class and
/// re-serialising that class over the `.inf_act`, and `emit ∘ parse` is a fixed
/// point on the TEXT (roundtrip law 2) rather than on the JSON: a graph-lowered
/// class carries synthetic local ids that renumber into the parser's walk
/// order, so a save would rewrite bytes the author never touched. Making that
/// safe is a decision about identity, not a wiring job.
///
/// `raise`-to-a-graph refuses far more than this does (a `NodeDef`'s ports are
/// fixed and a call form has none), which is the arc's own claim made
/// operational: **the text face refuses strictly less than the graph face**, so
/// a handler that cannot be drawn can still be read. Measured, over the shape
/// the bridge is sold on, by `a_handler_the_canvas_cannot_draw_still_reads_as_text`.
///
/// "Total over the IR" is the arc's shorthand and it is a shorthand: `emit_class`
/// has four refusal kinds (SCRIPT1a's memo prices them — four IR states no
/// producer makes, a depth bound, and one statement shape `raise` refuses too),
/// which is exactly why this function has an error arm at all. The `Err` below
/// is reachable, not defensive.
#[tauri::command]
pub async fn script_emit_class(
    asset_id: String,
    assets: tauri::State<'_, super::assets::AssetState>,
) -> Result<String, String> {
    let id: inf_asset::AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let class = assets.load_blueprint_class_result(id)?;
    inf_script::emit_class(&class).map_err(|e| {
        format!(
            "“{}” has no written form as InfiniScript ({e}); open the generated \
             code instead.",
            class.name
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The wire carries Ring 0's own numbers.**
    ///
    /// Asserted against `inf_script::render` — the string the CLI prints and the
    /// Output Log shows — rather than against a literal, because the claim worth
    /// making is that the panel and the compiler say the same thing about the
    /// same file, not that a `u32` survived serde. `render` writes `line:col:`
    /// through `Span`'s `Display`, so finding that prefix in it proves the DTO
    /// did not shift either number on the way out.
    #[test]
    fn a_refusals_line_and_column_cross_the_wire_intact() {
        // Line 3, and the `+` has nothing to add to.
        let src = "actor \"Wire\"\n\non tick(dt)\n  local x = 1 +\nend\n";
        let refusals = check_text(src, Some("C:/Content/Scripts/Wire.infini"));
        assert!(
            !refusals.is_empty(),
            "a truncated expression must refuse, or this arm is vacuous"
        );

        let rendered = inf_script::render(
            &inf_script::compile(src, "check:Wire.infini")
                .expect_err("this source does not compile"),
        );
        for d in &refusals {
            assert!(
                rendered.contains(&format!("{}:{}:", d.line, d.col)),
                "the DTO says {}:{} and Ring 0's own rendering says:\n{rendered}",
                d.line,
                d.col
            );
            assert!(
                rendered.contains(&d.message),
                "the DTO's message is not Ring 0's:\n  dto: {}\n  ring 0:\n{rendered}",
                d.message
            );
            assert_eq!(d.severity, "error");
            // 1-based, both of them — a 0 here would be an off-by-one that the
            // frontend's own conversion would then double.
            assert!(d.line >= 1 && d.col >= 1, "the span is 1-based");
        }
    }

    /// A file that compiles answers with its **warnings**, not with an error.
    ///
    /// The bare `undeclared` is a member variable by rule 3 of A.3 (anything
    /// that is not a local or a parameter is one), and a unit that declares no
    /// such `var` gets a warning rather than a refusal — so this is the one
    /// shape that proves `Ok` and "the list is empty" are different answers.
    #[test]
    fn a_script_that_compiles_can_still_have_something_to_say() {
        let clean = check_text(
            "actor \"Quiet\"\n\nvar hp: float = 1.0\n\non tick(dt)\n  hp = hp - dt\nend\n",
            None,
        );
        assert_eq!(clean, vec![], "a clean script has nothing to report");

        let warned = check_text(
            "actor \"Loud\"\n\non tick(dt)\n  undeclared = dt\nend\n",
            None,
        );
        assert_eq!(warned.len(), 1, "expected one warning, got {warned:?}");
        assert_eq!(warned[0].severity, "warning");
        assert!(
            warned[0].message.contains("undeclared"),
            "a warning must name the variable: {}",
            warned[0].message
        );
    }

    /// **The editor refuses what the cook refuses.** `compile_bytes` is the file
    /// door, so the 1 MiB ceiling applies to a buffer as well as to a file — and
    /// the message names the label, which is why `path` is carried at all.
    #[test]
    fn the_source_ceiling_applies_to_a_buffer_too() {
        let huge = "-- pad\n".repeat(inf_script::MAX_SOURCE_BYTES / 7 + 1);
        assert!(huge.len() > inf_script::MAX_SOURCE_BYTES);
        let refusals = check_text(&huge, Some("Content/Scripts/Huge.infini"));
        assert_eq!(refusals.len(), 1);
        assert!(
            refusals[0].message.contains("Huge.infini"),
            "the refusal must name the file the author is looking at: {}",
            refusals[0].message
        );
    }

    /// **What `script_emit_class` shows, `script_check` accepts.**
    ///
    /// The two commands in this module are the two halves of one loop — open a
    /// class as text, and the editor immediately lints that text — so the arm
    /// runs them against each other rather than each against a fixture.
    /// Ring 0 proves `parse(emit(f)) == f` on the IR (`roundtrip.rs` law 1);
    /// what is proved here is the consequence for the EDITOR: the buffer the
    /// author is shown does not arrive covered in squiggles.
    ///
    /// The class comes from the script every new project ships, so this is a
    /// claim about real content rather than about three statements I chose.
    #[test]
    fn the_text_a_class_opens_as_is_text_the_checker_accepts() {
        let source = include_str!("../../../../../templates/scripts/Example.infini");
        let (class, _warnings) =
            inf_script::compile(source, "check:Example").expect("the shipped template compiles");

        let text = inf_script::emit_class(&class).expect("a class the emitter can write");
        assert!(
            text.contains("on tick"),
            "the emitted text is not empty: {text}"
        );

        let refusals = check_text(&text, Some("Example.infini"));
        assert_eq!(
            refusals,
            vec![],
            "the editor would open this class covered in squiggles:\n{text}"
        );
    }

    /// **A handler the canvas cannot draw still reads as text** — the claim
    /// "Open as InfiniScript" exists for, measured rather than repeated.
    ///
    /// It is stated in three places (this module's doc, the drawer's context
    /// menu, the book's "Reading a Blueprint as InfiniScript") and until this
    /// arm it was measured in none: the only emit→check arm runs over
    /// `Example.infini`, which holds no unit-local call and therefore never
    /// crosses the asymmetry it is cited for. So the three faces are run
    /// against each other over one class that does:
    ///
    /// * the GRAPH face refuses it **by name** — `RaiseError::LocalFunctionCall`,
    ///   because a `NodeDef`'s ports are fixed at registration and a user
    ///   function's signature is not, so the palette has no node to draw it with;
    /// * the TEXT face writes it anyway;
    /// * `script_check` accepts what was written, so the tab does not open
    ///   covered in squiggles.
    ///
    /// Note what is NOT claimed: the emitter is not total. `emit_class` has four
    /// refusal kinds and `script_emit_class` renders one into a message naming
    /// the class. What is true, and what this measures, is the *direction*: the
    /// text face refuses strictly less than the graph face on the shape the
    /// bridge is sold on.
    #[test]
    fn a_handler_the_canvas_cannot_draw_still_reads_as_text() {
        let source = r#"actor "Bridge"

var out: float = 0.0

function bump(x: float)
  out = out + x
end

on begin_play()
  bump(21.0)
end
"#;
        let (class, _w) = inf_script::compile(source, "bridge").expect("the fixture compiles");
        let handler = &class
            .events
            .first()
            .expect("the fixture declares one handler")
            .body;

        // 1. The graph face refuses it, by the name the arc gave the refusal.
        let refused = inf_blueprint::raise_fn(handler)
            .err()
            .expect("`raise` must refuse a call form, or this arm is about nothing");
        assert!(
            matches!(&refused, inf_blueprint::RaiseError::LocalFunctionCall(n) if n == "bump"),
            "expected the call form's own refusal, got {refused:?}"
        );

        // 2. The text face writes it…
        let text = inf_script::emit_class(&class).expect("the emitter writes a call form");
        assert!(
            text.contains("bump(21.0)"),
            "the call form did not survive the emitter:\n{text}"
        );

        // 3. …and the checker accepts what was written, which is what the
        //    read-only tab's own linter does the moment it opens.
        assert_eq!(
            check_text(&text, Some("Bridge.infini")),
            vec![],
            "the editor would open this class covered in squiggles:\n{text}"
        );
    }

    /// The label is a leaf on both separators, and an unsaved buffer still has
    /// a name a message can use.
    #[test]
    fn a_label_is_a_leaf_or_a_stand_in() {
        assert_eq!(
            label_of(Some("C:/Content/Scripts/Door.infini")),
            "Door.infini"
        );
        assert_eq!(
            label_of(Some("C:\\Content\\Scripts\\Door.infini")),
            "Door.infini"
        );
        assert_eq!(label_of(Some("Door.infini")), "Door.infini");
        assert_eq!(label_of(None), "(unsaved script)");
        assert_eq!(label_of(Some("")), "(unsaved script)");
    }
}
