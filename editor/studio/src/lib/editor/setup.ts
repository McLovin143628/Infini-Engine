/**
 * CodeMirror 6 editor assembly (P5.1): the base extension set + the shared
 * `Compartment`s.
 *
 * **LSP seam (P5.2):** `extraCompartment` is reserved for the language layer.
 * The LSP integration installs its completion source, hover tooltip, and
 * diagnostics by reconfiguring it via `setExtraExtensions(view, ext)` — it never
 * has to rewrite EditorPanel. `languageCompartment` holds the per-file language.
 */
import {
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
} from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching, foldGutter, foldKeymap, indentOnInput } from "@codemirror/language";
import { lintKeymap } from "@codemirror/lint";
import { highlightSelectionMatches, search, searchKeymap } from "@codemirror/search";
import { Compartment, EditorState, type Extension } from "@codemirror/state";
import {
  drawSelection,
  dropCursor,
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
  rectangularSelection,
  type ViewUpdate,
} from "@codemirror/view";
import { indentationMarkers } from "@replit/codemirror-indentation-markers";
import { showMinimap } from "@replit/codemirror-minimap";

import { editorHighlighting, editorTheme } from "./cmTheme";
import { languageExtensionFor } from "./languages";

/** Per-file language extension (reconfigured on open). */
export const languageCompartment = new Compartment();
/** Reserved for the LSP layer (P5.2). Starts empty. */
export const extraCompartment = new Compartment();

/** Save handler (Ctrl/Cmd+S). Registered by the editor store to avoid an import
 *  cycle; defaults to a no-op until then. */
let saveHandler: () => void = () => {};
export function registerSaveHandler(fn: () => void): void {
  saveHandler = fn;
}

function minimap(): Extension {
  return showMinimap.compute([], () => ({
    create: () => ({ dom: document.createElement("div") }),
    displayText: "blocks",
    showOverlay: "always",
  }));
}

/** The full base extension set for a document at `path`. */
export function baseExtensions(path: string): Extension[] {
  return [
    lineNumbers(),
    highlightActiveLineGutter(),
    foldGutter(),
    drawSelection(),
    dropCursor(),
    EditorState.allowMultipleSelections.of(true),
    indentOnInput(),
    bracketMatching(),
    closeBrackets(),
    autocompletion(),
    rectangularSelection(),
    highlightActiveLine(),
    highlightSelectionMatches(),
    search(),
    history(),
    indentationMarkers(),
    minimap(),
    editorTheme(),
    editorHighlighting(),
    languageCompartment.of(languageExtensionFor(path) ?? []),
    extraCompartment.of([]),
    keymap.of([
      { key: "Mod-s", preventDefault: true, run: () => (saveHandler(), true) },
      ...closeBracketsKeymap,
      ...defaultKeymap,
      ...searchKeymap,
      ...historyKeymap,
      ...foldKeymap,
      ...completionKeymap,
      ...lintKeymap,
      indentWithTab,
    ]),
  ];
}

/** Build an EditorState for `doc` at `path`, with an update listener. */
export function createEditorState(
  doc: string,
  path: string,
  onUpdate: (u: ViewUpdate) => void,
): EditorState {
  return EditorState.create({
    doc,
    extensions: [...baseExtensions(path), EditorView.updateListener.of(onUpdate)],
  });
}

/** Reconfigure the LSP compartment for a live view (P5.2 entry point). */
export function setExtraExtensions(view: EditorView, ext: Extension): void {
  view.dispatch({ effects: extraCompartment.reconfigure(ext) });
}
