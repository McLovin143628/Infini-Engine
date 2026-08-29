//! **Refusals are values** (P21's law), and a refusal names a line, a column and
//! a remedy.
//!
//! The table below is the contract: for each broken source, the exact place the
//! compiler points at and a phrase the message must contain. Two meta-arms keep
//! it from rotting — [`nothing_in_the_table_is_a_panic`] runs every entry and
//! [`no_prefix_of_a_real_script_ever_panics`] feeds the parser every truncation
//! of a working file, which is what an editor does on every keystroke.

use inf_script::{compile, parse_fn, render, Severity};

/// `(what is wrong, source, line, column, a phrase the message must contain)`.
const BAD: &[(&str, &str, u32, u32, &str)] = &[
    (
        "a missing `end`",
        "on tick(dt)\n    debug.print(\"x\")\n",
        3,
        1,
        "expected `end`",
    ),
    (
        "an unknown namespace",
        "on tick(dt)\n    wibble.print(\"x\")\nend\n",
        2,
        5,
        "there is no `wibble` namespace",
    ),
    (
        "an unknown verb in a real namespace",
        "on tick(dt)\n    math.sinn(1.0)\nend\n",
        2,
        5,
        "the `math` namespace has no verb `sinn`",
    ),
    (
        "an operator written as a call",
        "on tick(dt)\n    local a = math.add(1, 2)\nend\n",
        2,
        15,
        "write `a + b`",
    ),
    (
        "control flow written as a call",
        "on tick(dt)\n    flow.branch(true)\nend\n",
        2,
        5,
        "if … then … end",
    ),
    (
        "a missing required argument",
        "on tick(dt)\n    engine.set_rotation()\nend\n",
        2,
        5,
        "needs its `angle` argument",
    ),
    (
        "too many arguments",
        "on tick(dt)\n    debug.print(\"a\", \"b\")\nend\n",
        2,
        5,
        "takes 1 argument (`message`), not 2",
    ),
    (
        "a multi-result query used without naming a result",
        "on tick(dt)\n    local h = physics2d.raycast(0.0, 0.0, 1.0, 0.0, 1.0)\nend\n",
        2,
        15,
        "returns several results",
    ),
    (
        "an unknown event",
        "on wibble()\nend\n",
        1,
        4,
        "`wibble` is not an event",
    ),
    (
        "an event handler with the wrong parameters",
        "on tick(t)\nend\n",
        1,
        8,
        "`(dt: float)`",
    ),
    (
        "`on input` without its action name",
        "on input(pressed)\nend\n",
        1,
        4,
        "needs a name",
    ),
    (
        "assigning to a parameter",
        "on tick(dt)\n    dt = 1.0\nend\n",
        2,
        5,
        "is a handler parameter and cannot be assigned",
    ),
    (
        "a shadowing local",
        "on tick(dt)\n    local a = 1\n    local a = 2\nend\n",
        3,
        11,
        "already a local here",
    ),
    (
        "a local shadowing a parameter",
        "on tick(dt)\n    local dt = 1\nend\n",
        2,
        11,
        "is a parameter of this handler",
    ),
    (
        "a chained comparison",
        "on tick(dt)\n    local a = 1 < 2 < 3\nend\n",
        2,
        21,
        "comparisons do not chain",
    ),
    (
        "a bare call with no namespace",
        "on tick(dt)\n    print(\"x\")\nend\n",
        2,
        5,
        "a call names a namespace and a verb",
    ),
    (
        "an unterminated string",
        "on tick(dt)\n    debug.print(\"x)\nend\n",
        2,
        17,
        "may not span lines",
    ),
    (
        "an unknown escape",
        "on tick(dt)\n    debug.print(\"a\\q\")\nend\n",
        2,
        19,
        "unknown escape",
    ),
    (
        "an unclosed `rust` block",
        "on tick(dt)\n    rust [[\n    let x = 1;\n",
        2,
        10,
        "unterminated",
    ),
    (
        "`rust` without a block",
        "on tick(dt)\n    rust \"oops\"\nend\n",
        2,
        10,
        "takes a `[[…]]` block",
    ),
    (
        "a variable default of the wrong type",
        "var speed: int = 1.5\n",
        1,
        18,
        "declared `int` but its default is a float",
    ),
    (
        "an unknown type",
        "var speed: number = 1\n",
        1,
        12,
        "`number` is not a type",
    ),
    (
        "a reserved local name",
        "on tick(dt)\n    local var = 1\nend\n",
        2,
        11,
        "is reserved",
    ),
    (
        "a reserved parameter spelling",
        "function f(n3: float)\nend\n",
        1,
        12,
        "is reserved",
    ),
    (
        "a function parameter with no type",
        "function f(x)\nend\n",
        1,
        12,
        "needs a type",
    ),
    (
        "two handlers for one event",
        "on tick(dt)\nend\non tick(dt)\nend\n",
        3,
        1,
        "already has a handler",
    ),
    (
        "an integer that does not fit",
        "on tick(dt)\n    local a = 99999999999999999999\nend\n",
        2,
        15,
        "does not fit a 64-bit integer",
    ),
    (
        "a stray character",
        "on tick(dt)\n    local a = @\nend\n",
        2,
        15,
        "unexpected character `@`",
    ),
    (
        "an `actor` header that is not first",
        "on tick(dt)\nend\nactor \"Late\"\n",
        3,
        1,
        "must be the file's first line",
    ),
];

#[test]
fn every_refusal_names_its_place_and_its_remedy() {
    // Every mismatch is collected rather than the first one panicking: a table
    // this size is edited in batches, and a gate that reports one row per run
    // costs a compile per row.
    let mut wrong: Vec<String> = Vec::new();
    for (what, src, line, col, phrase) in BAD {
        let diags = match compile(src, "act:bad") {
            Ok((_, warnings)) => {
                wrong.push(format!(
                    "{what}: compiled when it should have refused (warnings: {})",
                    render(&warnings)
                ));
                continue;
            }
            Err(d) => d,
        };
        let Some(first) = diags.iter().find(|d| d.severity == Severity::Error) else {
            wrong.push(format!("{what}: no error among {diags:?}"));
            continue;
        };
        if (first.span.line, first.span.col) != (*line, *col) {
            wrong.push(format!(
                "{what}: expected {line}:{col}, got {}:{} — `{}`",
                first.span.line, first.span.col, first.message
            ));
        }
        if !first.message.contains(phrase) {
            wrong.push(format!(
                "{what}: the message does not contain `{phrase}`: {}",
                first.message
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The table itself is the arm's falsifier: an entry that stops being wrong is
/// an entry nothing is checking.
#[test]
fn the_table_is_large_enough_to_cover_the_grammar() {
    assert!(
        BAD.len() >= 25,
        "only {} refusals are pinned; the grammar has more ways to be wrong",
        BAD.len()
    );
}

/// **Nothing panics.** The compiler is a library the editor calls on a
/// keystroke; a panic there takes the editor, not the script.
#[test]
fn nothing_in_the_table_is_a_panic() {
    for (_, src, ..) in BAD {
        let _ = compile(src, "act:bad");
        let _ = parse_fn(src);
    }
}

/// …and neither is any *prefix* of a working script, which is what a file looks
/// like halfway through being typed.
#[test]
fn no_prefix_of_a_real_script_ever_panics() {
    const GOOD: &str = r#"actor "Typing"

var angle: float = 0.0 exposed

on tick(dt)
    local a = angle + dt * 2.0
    if a > 1.0 then
        for i = 0, 3 do
            debug.print("x")
        end
    end
    angle = a
    rust [[
    let _ = 1;
]]
end
"#;
    let chars: Vec<char> = GOOD.chars().collect();
    let mut refused = 0;
    for cut in 0..=chars.len() {
        let prefix: String = chars[..cut].iter().collect();
        match compile(&prefix, "act:typing") {
            Ok(_) => {}
            Err(d) => {
                refused += 1;
                assert!(!d.is_empty(), "a refusal with no diagnostic at cut {cut}");
                // Every diagnostic points somewhere inside (or just past) the
                // text it was given — never at line 0, never past the end.
                for diag in &d {
                    assert!(diag.span.line >= 1, "line 0 at cut {cut}: {diag}");
                    assert!(
                        diag.span.line as usize <= prefix.lines().count() + 1,
                        "cut {cut}: {diag} is past the end of {} lines",
                        prefix.lines().count()
                    );
                }
            }
        }
    }
    assert!(
        refused > 100,
        "only {refused} of {} prefixes refused — the sweep is not exercising \
         the error paths",
        chars.len() + 1
    );
}

/// An undeclared member variable is a **warning**, not an error: the runtime
/// refuses it by name, which is P21's law, and stopping the compile would make
/// a whole handler unrunnable over one typo in one branch.
#[test]
fn an_undeclared_variable_warns_and_still_compiles() {
    let (class, warnings) = compile(
        "actor \"W\"\n\nvar speed: float = 1.0\n\non tick(dt)\n    debug.print(\"x\")\n    engine.set_rotation(spede)\nend\n",
        "act:w",
    )
    .expect("a warning does not stop a compile");
    assert_eq!(warnings.len(), 1, "{}", render(&warnings));
    assert_eq!(warnings[0].severity, Severity::Warning);
    assert_eq!((warnings[0].span.line, warnings[0].span.col), (7, 25));
    assert!(
        warnings[0].message.contains("`spede`") && warnings[0].message.contains("`speed`"),
        "the warning names the typo and what the file does declare: {}",
        warnings[0].message
    );
    assert_eq!(class.events.len(), 1);
}

/// A file with several broken declarations reports **all** of them, because an
/// editor that shows one error at a time makes a person compile once per typo.
#[test]
fn recovery_reports_every_broken_declaration() {
    let src = "on wibble()\nend\n\non tick(dt)\n    wobble.x()\nend\n\nvar speed: number = 1\n";
    let diags = compile(src, "act:many").unwrap_err();
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 3, "{}", render(&diags));
    assert_eq!(
        errors.iter().map(|d| d.span.line).collect::<Vec<_>>(),
        vec![1, 5, 8]
    );
}

/// `parse_fn` is the graph↔text bridge's door, so it refuses a source holding
/// none or several rather than silently taking the first.
#[test]
fn the_single_function_door_refuses_zero_and_many() {
    assert!(parse_fn("").is_err());
    assert!(parse_fn("on tick(dt)\nend\non begin_play()\nend\n").is_err());
    assert!(parse_fn("on tick(dt)\nend\n").is_ok());
    assert!(parse_fn("function f(x: float) -> float\n    return x\nend\n").is_ok());
}
