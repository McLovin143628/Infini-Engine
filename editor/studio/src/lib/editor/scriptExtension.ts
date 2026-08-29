/**
 * The InfiniScript editor layer (wave SCRIPT2b): Ring-0 refusals as inline
 * squiggles and Problems-panel rows.
 *
 * # One linter, one debounce, one source of truth
 *
 * `@codemirror/lint`'s `linter()` already owns the shape this needs — re-run
 * after a quiet period following a change, batch, and paint. So the check is a
 * lint source rather than an `updateListener` plus a hand-rolled timer, and the
 * same callback that paints the squiggles writes the Problems panel's row set.
 * Two timers over one buffer would eventually disagree about which answer is
 * current.
 *
 * The answer itself is never computed here. `script.check` hands the buffer to
 * `inf_script::compile_bytes` — the one file door the asset watcher, `inf cook`
 * and the PIE payload builder all enter — so the squiggle under a line and the
 * error in the shipped build are the same sentence from the same compiler.
 *
 * # Installed per DOCUMENT, not per active tab, and that is the hazard's fix
 *
 * A `Compartment` holds one configuration: `reconfigure` replaces. `lspBridge`
 * reconfigures `extraCompartment` on **every** active-tab change, so an
 * InfiniScript layer sharing that compartment would be evicted the next time
 * the author touched a `.rs` tab — silently, and only sometimes.
 *
 * Two things prevent it. First, `scriptCompartment` is the layer's own (the
 * third compartment). Second, and more decisive: this extension is a pure
 * function of the file's PATH, so `baseExtensions` installs it when the tab's
 * `EditorState` is built and no bridge ever reconfigures it on tab switch.
 * There is no live-view race to lose, and nothing to evict.
 *
 * # The honest bound
 *
 * A lint source runs in a live `EditorView`. A `.infini` tab that has never
 * been displayed has therefore never been checked, and contributes no rows to
 * the Problems panel — unlike rust-analyzer, which publishes for the whole
 * workspace. Saving the file is what tells the rest of the editor: the watcher
 * compiles it through the same door and the Output Log carries the result
 * (SCRIPT1b), and a `.infini` bound to an actor recompiles into a running
 * Simulate.
 */
import { linter, type Diagnostic } from "@codemirror/lint";
import type { Extension } from "@codemirror/state";
import type { EditorView } from "@codemirror/view";

import { script, type ScriptDiagnosticDto } from "../ipc";
import { useLspStore } from "../../stores/lspStore";
import { pathToUri } from "./fileUri";
import { diagnosticsToCM } from "./lspExtension";
import { scriptDiagnosticsToLsp } from "./scriptDiagnostics";

/**
 * How long the editor waits after the last keystroke before asking Ring 0.
 *
 * 250 ms, matching `commands/assets.rs`'s `WATCH_DEBOUNCE` — the same number
 * the file watcher waits before recompiling a saved script — so typing and
 * saving feel like one system rather than two with different reflexes. It is
 * exported because a test that hard-codes it would be a test about itself
 * (the SCRIPT1b lesson: a number a test picks is a number about the test).
 */
export const SCRIPT_CHECK_DEBOUNCE_MS = 250;

/** Is this path an InfiniScript source file? The extension test, once. */
export function isScriptPath(path: string): boolean {
  return path.toLowerCase().endsWith(".infini");
}

/**
 * The lint source for `path`: check the buffer, publish to the Problems panel,
 * and hand the squiggles back to CodeMirror.
 *
 * A failed IPC call answers with **no diagnostics** rather than a fabricated
 * one. The editor not knowing is not the same as the script being wrong, and
 * inventing an error at line 1 would be a refusal that is not true about the
 * program in front of it.
 */
export function checkSource(path: string) {
  const uri = pathToUri(path);
  return async (view: EditorView): Promise<Diagnostic[]> => {
    // The doc as it was when the check started. CodeMirror discards a result
    // whose document has moved on, so pinning it here is the honest offset base
    // rather than a race with `view.state`.
    const doc = view.state.doc;
    let refusals: ScriptDiagnosticDto[];
    try {
      refusals = await script.check(doc.toString(), path);
    } catch (e) {
      console.error("script_check failed", path, e);
      return [];
    }
    const lsp = scriptDiagnosticsToLsp(refusals);
    // **The panel obeys the same staleness rule as the gutter** (SCRIPT2b
    // audit). `lintPlugin` drops a result whose `state.doc` has moved on, and
    // two checks really can be in flight: the plugin schedules the next run the
    // moment the document changes, without waiting for the one already awaiting
    // IPC, and two `invoke`s have no ordering guarantee between them. Writing
    // the store unconditionally therefore made the Problems panel the one
    // surface that could sit showing refusals about a buffer that no longer
    // exists — and, if the replies crossed, showing the OLDER of two answers.
    // Skipping is safe because the run that supersedes this one publishes: the
    // linter always ends on a check whose document is still current.
    if (view.state.doc === doc) useLspStore.getState().setDiagnostics(uri, lsp);
    return diagnosticsToCM(doc, lsp);
  };
}

/**
 * The extension for the `scriptCompartment` of a document at `path` — the
 * linter for a `.infini`, nothing at all for anything else.
 */
export function scriptExtensionFor(path: string): Extension {
  if (!isScriptPath(path)) return [];
  return linter(checkSource(path), { delay: SCRIPT_CHECK_DEBOUNCE_MS });
}

/**
 * Drop a script's rows from the Problems panel.
 *
 * Called when its tab closes: a linter only exists inside a live view, so
 * without this the last refusals of a file nobody has open any more would sit
 * in the panel until the session ended. rust-analyzer's rows are the server's
 * to retract and are deliberately left alone.
 */
export function clearScriptDiagnostics(path: string): void {
  if (!isScriptPath(path)) return;
  useLspStore.getState().setDiagnostics(pathToUri(path), []);
}
