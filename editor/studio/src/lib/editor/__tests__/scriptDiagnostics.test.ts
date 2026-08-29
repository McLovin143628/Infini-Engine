/**
 * The InfiniScript diagnostics wire, on the frontend side (wave SCRIPT2b).
 *
 * The Rust half of this contract is `commands/script.rs`'s
 * `a_refusals_line_and_column_cross_the_wire_intact`, which pins the DTO
 * against `inf_script::render`'s own text. This half pins what happens to those
 * numbers once they arrive: **exactly one conversion**, from Ring 0's 1-based
 * line/column to the 0-based pair CodeMirror and the Problems panel speak.
 * An off-by-one here is a squiggle under the wrong word with nothing anywhere
 * saying so.
 */
import { Text } from "@codemirror/state";
import { describe, expect, it } from "vitest";

import { diagnosticsToCM } from "../lspExtension";
import { SCRIPT_CHECK_DEBOUNCE_MS, isScriptPath } from "../scriptExtension";
import { SCRIPT_DIAGNOSTIC_SOURCE, scriptDiagnosticsToLsp } from "../scriptDiagnostics";
import type { ScriptDiagnosticDto } from "../../ipc";

// Ring 2's own source, read rather than quoted (the SCRIPT1b lesson: a number a
// test picks is a number about the test).
import ASSETS_RS from "../../../../src-tauri/src/commands/assets.rs?raw";

const refusal = (over: Partial<ScriptDiagnosticDto> = {}): ScriptDiagnosticDto => ({
  severity: "error",
  line: 1,
  col: 1,
  len: 1,
  message: "something is wrong",
  ...over,
});

describe("InfiniScript refusals reaching the editor", () => {
  it("converts Ring 0's 1-based line and column to 0-based, once", () => {
    const [d] = scriptDiagnosticsToLsp([refusal({ line: 12, col: 5, len: 3 })]);
    expect(d.range.start).toEqual({ line: 11, character: 4 });
    expect(d.range.end).toEqual({ line: 11, character: 7 });
  });

  it("never produces a negative position, whatever arrives", () => {
    // Ring 0 spans are 1-based; a 0 would mean something upstream broke, and
    // the remedy is a range the document can hold rather than a crash.
    const [d] = scriptDiagnosticsToLsp([refusal({ line: 0, col: 0, len: 0 })]);
    expect(d.range.start).toEqual({ line: 0, character: 0 });
    expect(d.range.end.character).toBeGreaterThan(0);
  });

  it("widens a zero-length end-of-input span so it can be seen", () => {
    const [d] = scriptDiagnosticsToLsp([refusal({ line: 3, col: 1, len: 0 })]);
    expect(d.range.end.character - d.range.start.character).toBe(1);
  });

  it("maps the Ring-0 severity word onto the panel's number, and names its source", () => {
    expect(scriptDiagnosticsToLsp([refusal()])[0].severity).toBe(1);
    expect(scriptDiagnosticsToLsp([refusal({ severity: "warning" })])[0].severity).toBe(2);
    expect(scriptDiagnosticsToLsp([refusal()])[0].source).toBe(SCRIPT_DIAGNOSTIC_SOURCE);
  });

  it("lands on the document offsets the refusal names", () => {
    // Line 4, column 15 of this source is the `+` with nothing to add to.
    const src = ['actor "Wire"', "", "on tick(dt)", "  local x = 1 +", "end"].join("\n");
    const doc = Text.of(src.split("\n"));
    const [cm] = diagnosticsToCM(doc, scriptDiagnosticsToLsp([refusal({ line: 4, col: 15, len: 1 })]));
    expect(doc.sliceString(cm.from, cm.to)).toBe("+");
    expect(cm.severity).toBe("error");
    expect(cm.source).toBe(SCRIPT_DIAGNOSTIC_SOURCE);
  });

  it("recognises a script by its extension, case-insensitively", () => {
    expect(isScriptPath("C:/Content/Scripts/Door.infini")).toBe(true);
    expect(isScriptPath("C:/Content/Scripts/Door.INFINI")).toBe(true);
    expect(isScriptPath("C:/proj/src/main.rs")).toBe(false);
    expect(isScriptPath("infini")).toBe(false);
  });

  it("debounces at the interval the watcher debounces at", () => {
    // Typing and saving are one system; two different reflexes would be a
    // choice nobody made. `WATCH_DEBOUNCE` is read from `commands/assets.rs`
    // so this cannot drift into a number about itself.
    const watch = /WATCH_DEBOUNCE: Duration = Duration::from_millis\((\d+)\)/.exec(ASSETS_RS);
    expect(watch, "WATCH_DEBOUNCE is no longer declared in commands/assets.rs").not.toBeNull();
    expect(SCRIPT_CHECK_DEBOUNCE_MS).toBe(Number(watch?.[1]));
  });
});
