/**
 * LSP ⇆ editor bridge (P5.2). Owns the `lsp://` event subscriptions and drives
 * the editor: when a Rust file becomes active it ensures rust-analyzer is
 * running (rooted at the open project), sends didOpen, and installs the
 * per-file LSP extension into the editor's `extraCompartment` seam; incoming
 * diagnostics update the Problems store and, for the active file, paint
 * squiggles. Kept out of both stores to avoid an import cycle.
 */
import {
  onLspDiagnostics,
  onLspStarted,
  onLspStopped,
  refCountedInit,
  type UnlistenFn,
} from "../events";
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

/**
 * Start the bridge: subscribe to events + active-tab changes.
 *
 * **Refcounted, not flag-guarded** (F-lens L7.M2 — this was the LSP-is-dead
 * bug). The previous shape was `if (inited) return () => {}; inited = true;`,
 * which looks StrictMode-safe because the guard is set before the first
 * `await`. It is not, because of what the *second* caller is handed. React runs
 * mount → cleanup → mount synchronously in one commit:
 *
 *  1. mount #1 sets `inited` and starts subscribing;
 *  2. cleanup #1 marks itself disposed and arms `.then((fn) => fn())`;
 *  3. mount #2 sees `inited` still true (nothing has resolved) and receives a
 *     **no-op disposer** — it holds nothing;
 *  4. the subscriptions resolve, cleanup #1's armed disposer fires and tears
 *     down every `lsp://` listener plus the editor-store subscription, and
 *     resets `inited = false`.
 *
 * Net result in `tauri dev`: no listeners, no re-init, **no LSP at all** — no
 * diagnostics, no squiggles, and `activateForActive` never runs again, so
 * rust-analyzer is never even started. Counting holders instead of flipping a
 * flag makes step 3 a real second holder, so step 4 decrements to 1 rather than
 * to 0; and when the last holder does release, `refCountedInit` clears its
 * memo so a genuine remount re-subscribes.
 *
 * `started`/`openedUris`/`installedForPath` above are deliberately NOT reset by
 * the teardown: they describe the *backend server's* state, which outlives this
 * bridge's listeners.
 */
export const initLsp = refCountedInit(async () => {
  const disposers: UnlistenFn[] = [];

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
  const unsubscribeStore = useEditorStore.subscribe((s) => {
    if (s.activeId !== lastActive) {
      lastActive = s.activeId;
      void activateForActive();
    }
  });
  // Handle a file already open at init.
  void activateForActive();

  return () => {
    disposers.forEach((d) => d());
    disposers.length = 0;
    unsubscribeStore();
  };
});
