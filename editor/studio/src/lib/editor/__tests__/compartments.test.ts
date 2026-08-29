// @vitest-environment jsdom
/**
 * **THE THIRD COMPARTMENT** (wave SCRIPT2b), and the hazard it exists for.
 *
 * A CodeMirror `Compartment` holds exactly ONE configuration — `reconfigure`
 * replaces, it does not add — and `lspBridge.ts` reconfigures `extraCompartment`
 * on every active-tab change. So an InfiniScript layer sharing that compartment
 * would be evicted the next time the author touched a `.rs` tab: silently, and
 * only sometimes.
 *
 * This suite drives a real `EditorView` through the scouted sequence — open a
 * `.rs` tab, open a `.infini` tab, go back — with the panel's own state-swapping
 * (`EditorPanel.tsx`: save the outgoing state, `view.setState` the incoming one)
 * and the LSP bridge's own installer, and requires **both layers to survive**.
 */
import { forceLinting } from "@codemirror/lint";
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  baseExtensions,
  createEditorState,
  extraCompartment,
  languageCompartment,
  scriptCompartment,
  setExtraExtensions,
} from "../setup";

const RS = "C:/proj/src/main.rs";
const INFINI = "C:/proj/Content/Scripts/Door.infini";

/** Is a compartment's content non-empty in this state? */
function occupied(compartment: { get(state: EditorState): unknown }, state: EditorState): boolean {
  const held = compartment.get(state);
  return held !== undefined && !(Array.isArray(held) && held.length === 0);
}

describe("the editor's compartments", () => {
  // The linter fires `script_check` after its debounce; answer it so nothing
  // here can reach a real `invoke`.
  beforeEach(() => mockIPC(() => []));
  afterEach(() => clearMocks());

  it("gives InfiniScript a compartment of its own, filled from the path", () => {
    const rs = createEditorState("fn main() {}", RS, () => {});
    const infini = createEditorState('actor "Door"\n', INFINI, () => {});

    // Both get a language…
    expect(occupied(languageCompartment, rs)).toBe(true);
    expect(occupied(languageCompartment, infini)).toBe(true);
    // …only the script gets a linter…
    expect(occupied(scriptCompartment, infini)).toBe(true);
    expect(occupied(scriptCompartment, rs)).toBe(false);
    // …and neither starts with an LSP layer, which a bridge installs later.
    expect(occupied(extraCompartment, rs)).toBe(false);
    expect(occupied(extraCompartment, infini)).toBe(false);
  });

  it("keeps both layers across .rs → .infini → .rs", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const view = new EditorView({ parent: host });

    const rsState = createEditorState("fn main() {}", RS, () => {});
    const infiniState = createEditorState('actor "Door"\n', INFINI, () => {});

    // 1. The .rs tab is active and the LSP bridge installs its layer, exactly as
    //    `activateForActive` does.
    view.setState(rsState);
    setExtraExtensions(view, EditorView.editable.of(true));
    expect(occupied(extraCompartment, view.state)).toBe(true);
    const savedRs = view.state; // EditorPanel persists the outgoing state

    // 2. Switch to the .infini tab. Its own state carries the linter, and the
    //    LSP bridge — which early-returns for a non-Rust path — leaves this
    //    tab's `extraCompartment` alone.
    view.setState(infiniState);
    expect(occupied(scriptCompartment, view.state)).toBe(true);
    expect(occupied(extraCompartment, view.state)).toBe(false);
    const savedInfini = view.state;

    // 3. Back to the .rs tab. THE ARM: the LSP layer came back with its state,
    //    and the script layer is still on the state it belongs to.
    view.setState(savedRs);
    expect(occupied(extraCompartment, view.state)).toBe(true);
    expect(occupied(scriptCompartment, view.state)).toBe(false);
    expect(occupied(scriptCompartment, savedInfini)).toBe(true);

    view.destroy();
    host.remove();
  });

  it("would have been evicted had the two shared one compartment", () => {
    // The counterfactual, run rather than argued: put a script layer into
    // `extraCompartment` — the share the scouting warned about — and let the LSP
    // bridge do the one thing it does on every tab change.
    const view = new EditorView({ state: createEditorState("", INFINI, () => {}) });
    view.dispatch({ effects: extraCompartment.reconfigure(EditorView.editable.of(true)) });
    expect(occupied(extraCompartment, view.state)).toBe(true);
    setExtraExtensions(view, []); // the .rs tab arrives with nothing to install
    expect(occupied(extraCompartment, view.state)).toBe(false);
    // …while the real seam survives the same event untouched.
    expect(occupied(scriptCompartment, view.state)).toBe(true);
    view.destroy();
  });

  it("puts the script layer in the base set, so no bridge has to remember", () => {
    // `baseExtensions` is what `createEditorState` spreads; a future consumer
    // that installs the linter from an effect instead would re-open the hazard.
    //
    // The audit's fix: this arm compared the two arrays' LENGTHS, which are
    // equal whether or not the compartment holds anything — `scriptCompartment`
    // is one element of the array either way. It now builds a state out of
    // `baseExtensions` ALONE, with no `createEditorState` and no bridge, and
    // asks the compartment what it holds.
    const fromBase = (path: string) => EditorState.create({ extensions: baseExtensions(path) });
    expect(occupied(scriptCompartment, fromBase(INFINI))).toBe(true);
    expect(occupied(scriptCompartment, fromBase(RS))).toBe(false);
    // …and the language layer travels the same way, which is the property that
    // makes both a pure function of the path.
    expect(occupied(languageCompartment, fromBase(INFINI))).toBe(true);
  });

  /**
   * **`.infini` → `.infini`, which the first matrix never walked** (audit).
   *
   * The three-step matrix above switches between two *kinds* of file. The case
   * a per-path extension can actually get wrong is two files of the SAME kind:
   * if the linter were installed by a bridge watching the active tab, or shared
   * between documents, the second script would be checked under the first's
   * path — and every refusal in it would be filed against the wrong file in the
   * Problems panel, which is silent and permanent.
   *
   * `forceLinting` runs the source without waiting out the debounce, so this is
   * a claim about the wiring and not about a timer.
   */
  it("checks each script under its own path, across a .infini → .infini switch", async () => {
    const calls: { text: string; path: string }[] = [];
    mockIPC((cmd, args) => {
      if (cmd === "script_check") calls.push(args as unknown as { text: string; path: string });
      return [];
    });
    const host = document.createElement("div");
    document.body.appendChild(host);
    const view = new EditorView({ parent: host });

    const A = "C:/proj/Content/Scripts/A.infini";
    const B = "C:/proj/Content/Other/B.infini"; // …and not under Scripts/, either
    const a = createEditorState('actor "A"\n', A, () => {});
    const b = createEditorState('actor "B"\n', B, () => {});

    view.setState(a);
    forceLinting(view);
    await vi.waitFor(() => expect(calls.length).toBe(1));
    view.setState(b);
    forceLinting(view);
    await vi.waitFor(() => expect(calls.length).toBe(2));
    // …and back, on the saved state, which is what the panel does.
    view.setState(a);
    forceLinting(view);
    await vi.waitFor(() => expect(calls.length).toBe(3));

    expect(calls.map((c) => c.path)).toEqual([A, B, A]);
    expect(calls.map((c) => c.text)).toEqual(['actor "A"\n', 'actor "B"\n', 'actor "A"\n']);
    // Each document carries its own linter instance rather than a shared one.
    expect(scriptCompartment.get(a)).not.toBe(scriptCompartment.get(b));

    view.destroy();
    host.remove();
  });
});
