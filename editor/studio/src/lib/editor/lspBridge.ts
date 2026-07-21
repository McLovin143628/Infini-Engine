/**
 * LSP ⇆ editor bridge (P5.2). Owns the `lsp://` event subscriptions and drives
 * the editor: when a Rust file becomes active it ensures rust-analyzer is
 * running (rooted at the open project), sends didOpen, and installs the
 * per-file LSP extension into the editor's `extraCompartment` seam; incoming
 * diagnostics update the Problems store and, for the active file, paint
 * squiggles. Kept out of both stores to avoid an import cycle.
 */
import { onLspDiagnostics, onLspStarted, onLspStopped, type UnlistenFn } from "../events";
import { lsp } from "../ipc";
import { useProjectStore } from "../../stores/projectStore";
import {
  activeTabPath,
  getActiveView,
  useEditorStore,
} from "../../stores/editorStore";
import { useLspStore } from "../../stores/lspStore";
import { setExtraExtensions } from "./setup";
import { lspExtensionFor, pushDiagnostics } from "./lspExtension";

const LANG = "rust";

/** Match the backend `path_to_uri` (Windows drive paths → file:///C:/…). */
export function pathToUri(path: string): string {
  const p = path.replace(/\\/g, "/").replace(/ /g, "%20");
  return p.startsWith("/") ? `file://${p}` : `file:///${p}`;
}

function isRust(path: string | null): path is string {
  return !!path && path.toLowerCase().endsWith(".rs");
}

let started = false;
const openedUris = new Set<string>();
let installedForPath: string | null = null;

/** Ensure the server is up and the active Rust file is wired to the editor. */
async function activateForActive(): Promise<void> {
  const path = activeTabPath();
  const view = getActiveView();
  if (!isRust(path) || !view) return;

  const root = useProjectStore.getState().current?.root;
  if (!root) return; // no project → nothing to root the server at

  const uri = pathToUri(path);
  const lspApi = useLspStore.getState();

  // Start rust-analyzer once, rooted at the project.
  if (!started) {
    started = true;
    lspApi.setStatus(LANG, "starting");
    try {
      await lsp.start(LANG, root);
      lspApi.setStatus(LANG, "running");
    } catch (e) {
      started = false;
      lspApi.setStatus(LANG, "error");
      lspApi.setError(String(e));
      return;
    }
  }

  // didOpen once per file.
  if (!openedUris.has(uri)) {
    openedUris.add(uri);
    const state = view.state;
    void lsp.didOpen(LANG, path, state.doc.toString());
  }

  // Install the per-file LSP extension into the live view (idempotent per path).
  if (installedForPath !== path) {
    installedForPath = path;
    setExtraExtensions(view, lspExtensionFor({ path, uri }));
  }

  // Paint any diagnostics we already have for this file.
  const known = useLspStore.getState().diagnostics[uri];
  if (known) pushDiagnostics(view, known);
}

let disposers: UnlistenFn[] = [];
let unsubscribeStore: (() => void) | null = null;
let inited = false;

/** Start the bridge: subscribe to events + active-tab changes. Idempotent. */
export async function initLsp(): Promise<() => void> {
  if (inited) return () => {};
  inited = true;

  disposers.push(
    await onLspStarted((p) => useLspStore.getState().setStatus(p.language, "running")),
  );
  disposers.push(
    await onLspStopped((p) => useLspStore.getState().setStatus(p.language, "stopped")),
  );
  disposers.push(
    await onLspDiagnostics(({ uri, diagnostics }) => {
      useLspStore.getState().setDiagnostics(uri, diagnostics);
      const path = activeTabPath();
      const view = getActiveView();
      if (view && path && pathToUri(path) === uri) pushDiagnostics(view, diagnostics);
    }),
  );

  // React to the active tab changing.
  let lastActive: string | null = null;
  unsubscribeStore = useEditorStore.subscribe((s) => {
    if (s.activeId !== lastActive) {
      lastActive = s.activeId;
      void activateForActive();
    }
  });
  // Handle a file already open at init.
  void activateForActive();

  return () => {
    disposers.forEach((d) => d());
    disposers = [];
    unsubscribeStore?.();
    unsubscribeStore = null;
    inited = false;
  };
}
