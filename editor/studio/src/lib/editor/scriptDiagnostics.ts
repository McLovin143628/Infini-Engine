/**
 * InfiniScript refusals → the shape the editor already understands (SCRIPT2b).
 *
 * `ScriptDiagnosticDto` carries Ring 0's own numbers: **1-based** line and
 * column, length in **characters**, severity as a word. The Problems panel and
 * the squiggles both speak the LSP shape (0-based line, utf-16 character,
 * integer severity), and `diagnosticsToCM` is deliberately source-agnostic —
 * so exactly one function converts, here, and everything downstream of it is
 * the machinery rust-analyzer already drives.
 *
 * The one interesting case is a **zero-length span**, which Ring 0 uses at end
 * of input: there is no text to underline, so the range is widened by one
 * character. Without that a "the file ends before `end`" refusal is an
 * invisible squiggle on an empty column.
 */
import type { LspDiagnostic } from "../events";
import type { ScriptDiagnosticDto } from "../ipc";

/** The `source` every InfiniScript diagnostic carries into the Problems panel. */
export const SCRIPT_DIAGNOSTIC_SOURCE = "infiniscript";

/** LSP severity numbers, spelled once. */
const SEVERITY_ERROR = 1;
const SEVERITY_WARNING = 2;

/** Convert Ring 0's refusals to the LSP shape the editor's panels consume. */
export function scriptDiagnosticsToLsp(diags: ScriptDiagnosticDto[]): LspDiagnostic[] {
  return diags.map((d) => {
    // 1-based → 0-based, on both axes, once.
    const line = Math.max(0, d.line - 1);
    const character = Math.max(0, d.col - 1);
    return {
      range: {
        start: { line, character },
        // A zero-length span is "here, at the end" — widen it so it can be seen.
        end: { line, character: character + Math.max(1, d.len) },
      },
      severity: d.severity === "warning" ? SEVERITY_WARNING : SEVERITY_ERROR,
      message: d.message,
      source: SCRIPT_DIAGNOSTIC_SOURCE,
    };
  });
}
